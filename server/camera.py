"""
Thin ``Camera`` proxy.

Each ``Camera`` represents one potential input device that the user can
select as ``scope_camera`` / ``spotter_cameraN``.  It owns **no** worker
process of its own — there is exactly one shared ``CameraWorkerProcess``
managed by :class:`~server.camerascontroller.CamerasController`.

Responsibilities of this proxy:

* Probe the hardware once (on ``__init__``) for capabilities / supported
  resolutions — everything else comes from settings.
* Keep the user-facing settings snapshot.
* Activate the singleton worker on the first stream access and deactivate
  it on :meth:`stop`.
* Forward pose samples received from the worker into
  :meth:`AIAgent.engage` and ship the resulting engagement snapshot back
  to the worker so it can draw the AI overlay (scope_camera only).
* Expose JPEG frames (:class:`~server.common.JpegFrame`) via
  :meth:`get_stream_frame`.

The proxy itself runs a lightweight background thread only for the
scope_camera pose-engagement round-trip.  For non-AI modes the HTTP
relay reads bytes directly from the controller.
"""

from __future__ import annotations

import time
import threading
from copy import deepcopy
from enum import Enum

from .common import JpegFrame, EPSILON_DELAY, _make_loading_jpeg
from .cameradevice import create_camera_device


class CameraType(Enum):
    CV2 = "cv2"
    RPI = "rpi"
    DUMMY = "dummy"
    UNKNOWN = "unknown"


class Camera(threading.Thread):
    """Thin proxy whose public API matches the previous Camera class."""

    TIMEOUT_SECONDS = 5
    STOP_TIMEOUT_SECONDS = 4.0

    def __init__(
        self,
        *,
        index: int,
        camera_index: int,
        camera_code: str | None = None,
        camera_name: str | None = None,
        camera_type: str = "dummy",
        settings: dict | None = None,
        master_controller: "MasterController" = None,  # pyright: ignore[reportUndefinedVariable]
    ) -> None:
        super().__init__()
        self.daemon = True

        self._master_controller = master_controller
        self._index = index
        self._camera_index = camera_index
        self._camera_code = camera_code
        self._camera_type_str = camera_type
        self._camera_type = (
            CameraType(camera_type) if camera_type in [e.value for e in CameraType]
            else CameraType.UNKNOWN
        )
        self._camera_name = camera_name or f"{index}: Loading, please wait a minute..."

        self._state_lock = threading.RLock()
        self._active = False

        # Last-access timestamps drive the demand flags sent to the worker.
        self._last_access_time_raw: float = 0.0
        self._last_access_time_masked: float = 0.0
        self._last_access_time_ai: float = 0.0
        self._last_access_lock = threading.Lock()

        # NOTE: per-mode "last sequence handed out" used to live here as
        # ``_last_seq_raw/masked/ai``.  That leaked state across HTTP
        # connections — a newly-mounted ``<img>`` would inherit a stale
        # ``since`` from an earlier session and sit on "no-change" until
        # the worker produced a fresh frame (e.g. after a mode switch +
        # Apply), leaving the browser blank or showing old pixels.  The
        # counter now lives in ``generate_camera_frames`` (per HTTP
        # connection) and is passed into ``get_stream_frame`` via
        # ``since_sequence``.

        self._stream_id = 0

        # --- Probe hardware for capabilities / resolutions ---
        probe_device = create_camera_device(
            camera_type=camera_type,
            camera_index=camera_index,
            camera_name=self._camera_name,
            settings=settings or {},
        )
        self.supported_resolutions = probe_device.get_supported_resolutions()
        self._capabilities = probe_device.get_capabilities()

        self._default_settings = probe_device.get_properties()
        self._default_settings["auto_focus"] = True
        self._default_settings["auto_white_balance_temperature"] = True
        self._default_settings["auto_exposure"] = True
        self._default_settings["index"] = index
        self._default_settings["camera_index"] = camera_index
        self._default_settings["camera_type"] = camera_type
        self._default_settings["name"] = self._camera_name
        self._default_settings["supported_resolutions"] = self.supported_resolutions
        probe_device.close()

        # Settings
        self._settings: dict | None = None
        self._settings_version = 0
        self._applied_settings_version = -1
        self._settings_modified = False

        camera_settings = deepcopy(self._default_settings)
        if settings is not None:
            camera_settings.update(settings)
        self.settings = camera_settings

        # JPEG quality requested by the most recent streaming consumer per mode.
        self._quality_raw = 80
        self._quality_masked = 80
        self._quality_ai = 80

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def camera_name(self) -> str:
        return self._camera_name

    @property
    def settings(self) -> dict:
        with self._state_lock:
            if self._settings is not None:
                return deepcopy(self._settings)
        return deepcopy(self._default_settings)

    @settings.setter
    def settings(self, value: dict | None):
        with self._state_lock:
            self._settings = deepcopy(value) if value is not None else None
            self._settings_version += 1
            self._settings_modified = self._applied_settings_version != self._settings_version

    @property
    def camera_code(self) -> str | None:
        with self._state_lock:
            return self._camera_code

    @camera_code.setter
    def camera_code(self, value: str | None):
        with self._state_lock:
            self._camera_code = value

    @property
    def capabilities(self) -> dict:
        return self._capabilities

    @property
    def camera_type_str(self) -> str:
        return self._camera_type_str

    @property
    def index(self) -> int:
        return self._index

    @property
    def physical_camera_index(self) -> int:
        return self._camera_index

    # ------------------------------------------------------------------
    # Settings helpers (used by controller when propagating to worker)
    # ------------------------------------------------------------------
    def _get_pending_settings_update(self) -> tuple[dict | None, int] | None:
        with self._state_lock:
            if self._applied_settings_version == self._settings_version:
                self._settings_modified = False
                return None
            snapshot = deepcopy(self._settings) if self._settings is not None else None
            return snapshot, self._settings_version

    def _mark_settings_applied(self, *, settings_version: int) -> None:
        with self._state_lock:
            if settings_version > self._applied_settings_version:
                self._applied_settings_version = settings_version
            self._settings_modified = self._applied_settings_version != self._settings_version

    def reset_settings(self):
        self.settings = deepcopy(self._default_settings)

    # ------------------------------------------------------------------
    # Stream token management
    # ------------------------------------------------------------------
    def create_stream_token(self) -> int:
        with self._state_lock:
            return self._stream_id

    def _invalidate_stream_tokens(self) -> None:
        with self._state_lock:
            self._stream_id += 1

    def _is_stream_token_current(self, *, stream_token: int | None) -> bool:
        if stream_token is None:
            return True
        with self._state_lock:
            return self._stream_id == stream_token

    # ------------------------------------------------------------------
    # Headless AI check
    # ------------------------------------------------------------------
    def _is_headless_ai_processing_required(self) -> bool:
        if self._camera_code != "scope_camera":
            return False
        mc = self._master_controller
        if mc is None:
            return False
        ai_agent = getattr(mc, "ai_agent", None)
        if ai_agent is None:
            return False
        fn = getattr(ai_agent, "is_fully_activated", None)
        if not callable(fn):
            return False
        try:
            return fn() is True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def _touch_access(self, *, mode: int) -> None:
        now = time.time()
        with self._last_access_lock:
            if mode == 3:
                self._last_access_time_ai = now
                self._last_access_time_masked = now
                self._last_access_time_raw = now
            elif mode == 1:
                self._last_access_time_masked = now
                self._last_access_time_raw = now
            else:
                self._last_access_time_raw = now
        # Push demand to the worker synchronously so the first stream
        # request does not have to wait for the proxy thread's periodic
        # update (which competes with _process_pose / AIAgent.engage).
        mc = self._master_controller
        controller = getattr(mc, "cameras_controller", None) if mc is not None else None
        if controller is not None:
            try:
                controller.update_demand_from_camera(self)
            except Exception:
                pass

    def get_last_access_times(self) -> tuple[float, float, float]:
        with self._last_access_lock:
            return (
                self._last_access_time_raw,
                self._last_access_time_masked,
                self._last_access_time_ai,
            )

    def get_requested_qualities(self) -> tuple[int, int, int]:
        return self._quality_raw, self._quality_masked, self._quality_ai

    def set_requested_quality(self, *, mode: int, quality: int) -> None:
        q = max(1, min(int(quality), 100))
        if mode == 3:
            self._quality_ai = q
        elif mode == 1:
            self._quality_masked = q
        else:
            self._quality_raw = q

    def _ensure_active(self, *, stream_token: int | None = None) -> bool:
        if not self._is_stream_token_current(stream_token=stream_token):
            return False

        mc = self._master_controller
        if mc is None:
            return False
        controller = getattr(mc, "cameras_controller", None)
        if controller is None:
            return False

        # Ask the controller to put the singleton worker on this camera.
        activated = controller.activate_camera(self)
        if not activated:
            return False

        if not self.is_alive():
            try:
                if not self._is_stream_token_current(stream_token=stream_token):
                    return False
                if not self.is_alive() and not self._active:
                    threading.Thread.__init__(self)
                    self.daemon = True
                self.start()
                # Shortened spin-wait: if the worker legitimately needs
                # more time to become active, the next get_stream_frame
                # call drives another wait; meanwhile the HTTP generator
                # immediately gets a Loading JPEG so the browser doesn't
                # sit on a blank <img>.
                for _ in range(int(0.5 / EPSILON_DELAY)):
                    if not self._is_stream_token_current(stream_token=stream_token):
                        return False
                    if self._active:
                        break
                    time.sleep(EPSILON_DELAY)
            except RuntimeError as err:
                print(f"Error starting camera proxy thread for {self._camera_name}: {err}")
                return False
        return self._is_stream_token_current(stream_token=stream_token)

    # How long the consumer briefly blocks waiting for the worker to
    # publish the next frame before returning a "no-change" sentinel.
    # Keep short so failed waits (worker idle) don't stall the relay.
    _WAIT_FOR_NEW_TIMEOUT = 0.030

    # Fallback chains per requested mode.  For modes 0 and 1 we walk
    # down to simpler rings when the primary is still at seq=0 so the
    # user sees *something* live instead of the loading placeholder.
    #
    # Mode 3 (AI) deliberately has NO fallback: the client must never
    # see a masked/raw stand-in when it asked for AI overlays, because
    # alternating between "no circles" and "circles" frames during the
    # YOLO warm-up window looks like flicker.  While the AI ring is
    # cold (seq_ai == 0, e.g. the ~10s YOLO first-inference window),
    # get_stream_frame returns a loading frame and the HTTP relay
    # emits the "Loading, please wait a minute..." JPEG (rate-limited
    # to 0.5s).  The stream switches to real AI frames as soon as the
    # worker publishes the first post-YOLO frame.
    _MODE_FALLBACK_CHAIN: dict[int, tuple[tuple[int, str], ...]] = {
        3: ((3, "ai"),),
        1: ((1, "masked"), (0, "raw")),
        0: ((0, "raw"),),
    }

    def get_stream_frame(
        self,
        *,
        mode: int,
        stream_token: int,
        quality: int | None = None,
        since_by_mode: dict[str, int] | None = None,
    ) -> JpegFrame | None:
        """Return the latest JPEG frame for *mode* captured by the shared
        worker.

        *since_by_mode* maps ``"raw" / "masked" / "ai"`` to the last
        sequence the caller already consumed for THAT ring.  The
        generator keeps this dict as a per-HTTP-connection local
        variable, which

          * eliminates cross-connection state leaks: a freshly-mounted
            browser ``<img>`` passes ``0`` for every key and always
            receives the newest ring frame (or a Loading placeholder)
            on its very first call; and
          * lets us fall back to a lower-mode ring independently — the
            caller tracks per-ring "last sent" cursors so e.g. we can
            stream masked frames while the AI ring warms up and then
            switch to AI without re-emitting frames the browser has
            already seen.

        The returned ``JpegFrame.mode`` tells the caller which ring
        actually served the frame so it can update the right bucket
        in its ``since_by_mode`` tracker.

        Return shapes:

        * ``None`` — stream token is stale (camera switched or stopped);
          HTTP relay should break out of its yield loop.
        * ``JpegFrame(valid=True, data=<bytes>, ...)`` — a brand-new JPEG
          from ring ``JpegFrame.mode``.  The relay updates
          ``since_by_mode[<key>]`` from ``frame.sequence`` and yields
          the bytes.
        * ``JpegFrame(valid=False, data=<loading_bytes>, ...)`` — real
          "Loading..." placeholder: every ring in the fallback chain
          for this *mode* is still at ``seq=0`` (or producing for
          another camera).
        * ``JpegFrame(valid=False, data=b"", ...)`` — "no-change"
          sentinel: at least one ring has frames but none of them has
          anything newer than what the caller already holds.  The HTTP
          relay should skip yielding and just retry.
        """
        if not self._is_stream_token_current(stream_token=stream_token):
            return None

        if quality is not None:
            self.set_requested_quality(mode=mode, quality=int(quality))

        if not self._ensure_active(stream_token=stream_token):
            return self._loading_frame(mode=mode)

        self._touch_access(mode=mode)

        mc = self._master_controller
        controller = getattr(mc, "cameras_controller", None) if mc is not None else None
        if controller is None:
            return self._loading_frame(mode=mode)

        since_map = since_by_mode or {}
        chain = self._MODE_FALLBACK_CHAIN.get(int(mode), self._MODE_FALLBACK_CHAIN[0])

        # Track whether we saw at least one ring with real frames (seq>0)
        # but no fresher bytes — so we can return the "no-change"
        # sentinel instead of the "Loading" placeholder.
        any_ring_has_frames = False

        for ring_mode, ring_key in chain:
            since = int(since_map.get(ring_key, 0) or 0)
            data, ts, frame_mode, frame_camera_index, seq = controller.read_jpeg_for(
                camera=self, mode_key=ring_key, since_sequence=since,
            )

            # Only the PRIMARY ring is worth briefly waiting on — for
            # fallback rings we prefer to poll and move on so the chain
            # stays snappy.  This is what keeps the AI ring from
            # adding latency once masked frames are already showing.
            if (
                ring_key == chain[0][1]
                and data is None
                and seq > 0
                and frame_camera_index == self._index
            ):
                if controller.wait_for_new_jpeg_for(
                    camera=self, mode_key=ring_key, timeout=self._WAIT_FOR_NEW_TIMEOUT,
                ):
                    data, ts, frame_mode, frame_camera_index, seq = controller.read_jpeg_for(
                        camera=self, mode_key=ring_key, since_sequence=since,
                    )

            if not self._is_stream_token_current(stream_token=stream_token):
                return None

            if seq > 0 and frame_camera_index == self._index:
                any_ring_has_frames = True

            if data is not None and seq > 0 and frame_camera_index == self._index:
                # Fresh frame on this ring — return immediately; lower
                # rings stay intentionally un-read so we don't burn
                # work or steal latency from the primary.
                return JpegFrame(
                    valid=True, data=data, time=ts, mode=frame_mode,
                    sequence=seq, camera_index=frame_camera_index,
                )

            # Ring empty or no-change — try the next one in the chain.
            _ = ring_mode  # silence "unused" — kept for debug clarity

        # Nothing fresh anywhere.  Pick the signal that matches reality:
        # if at least one ring exists but has no new bytes, the caller
        # should skip (no-change); if every ring is still at seq=0, the
        # worker genuinely hasn't produced anything yet, so show
        # Loading.
        if any_ring_has_frames:
            return self._no_change_frame(mode=mode)
        return self._loading_frame(mode=mode)

    def _loading_frame(self, *, mode: int) -> JpegFrame:
        return JpegFrame(
            valid=False, data=_make_loading_jpeg(), time=time.time(),
            mode=mode, sequence=0, camera_index=self._index,
        )

    def _no_change_frame(self, *, mode: int) -> JpegFrame:
        return JpegFrame(
            valid=False, data=b"", time=time.time(),
            mode=mode, sequence=0, camera_index=self._index,
        )

    # ------------------------------------------------------------------
    # Proxy thread — handles AI engagement for scope_camera and keeps the
    # camera "selected" on the worker as long as there is interest.
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._active = True
        mc = self._master_controller
        controller = getattr(mc, "cameras_controller", None) if mc is not None else None

        try:
            self._touch_access(mode=0)

            while self._active:
                if controller is None:
                    break

                # Still the active camera?  If the controller has switched
                # to a different camera we exit the proxy thread so the new
                # camera's proxy can take over.
                if not controller.is_camera_active(self):
                    break

                # Propagate settings changes for this camera to the worker.
                pending = self._get_pending_settings_update()
                if pending is not None:
                    snapshot, version = pending
                    if snapshot is not None:
                        try:
                            controller.update_camera_settings(self, snapshot)
                        except Exception as exc:
                            print(f"Error sending camera settings for {self._camera_name}: {exc}")
                    self._mark_settings_applied(settings_version=version)

                # Forward demand + quality to the worker BEFORE _process_pose
                # so that a slow AIAgent.engage iteration can't delay the
                # worker's reaction to a newly-requested mode (e.g. browser
                # just opened a Raw stream).
                controller.update_demand_from_camera(self)

                # Pose samples (scope_camera drives the AI engagement loop).
                self._process_pose(controller=controller)

                if not self._should_stay_alive():
                    break

                time.sleep(EPSILON_DELAY)

        except Exception as exc:
            print(f"Error in camera proxy thread ({self._camera_name}): {exc}")
        finally:
            self._active = False
            if controller is not None:
                try:
                    controller.deactivate_camera(self)
                except Exception as exc:
                    print(f"Error deactivating camera {self._camera_name}: {exc}")

    # ------------------------------------------------------------------
    # Pose / engagement
    # ------------------------------------------------------------------
    def _process_pose(self, *, controller) -> None:
        """Consume raw-pose samples pushed by the camera worker and run the
        engagement state machine here in the main process.

        Note on pose drawing: the camera worker now translates its **own**
        fresh ``raw_pose`` into a ``pose_dict`` in the same iteration it draws
        on — the ``pose_dict`` we send back through ``ai_result_pipe`` below
        is used only by :meth:`AIAgent.build_engagement_snapshot` (and for
        diagnostics), **not** for drawing the organ circles.  That split
        eliminates the 1-5 frame lag the circles used to have relative to
        the underlying motion.
        """
        if self._camera_code != "scope_camera":
            controller.drain_pose_for_non_ai(camera_index=self._index)
            return

        mc = self._master_controller
        ai_agent = getattr(mc, "ai_agent", None) if mc is not None else None
        if ai_agent is None:
            controller.drain_pose_for_non_ai(camera_index=self._index)
            return

        msgs = controller.collect_pose_messages(camera_index=self._index)
        if not msgs:
            # No pose messages queued by the worker — skip the entire
            # placeholder-Frame / AIAgent.engage path so the proxy loop
            # stays responsive (it needs to forward demand updates and
            # settings).
            return

        # Always process the latest sample only (engagement is stateful and
        # uses the newest information it has).
        latest = msgs[-1]
        raw_pose = latest.get("pose") if isinstance(latest, dict) else None
        frame_time = latest.get("time", time.time()) if isinstance(latest, dict) else time.time()
        image_width = int(latest.get("image_width", 0)) if isinstance(latest, dict) else 0
        image_height = int(latest.get("image_height", 0)) if isinstance(latest, dict) else 0

        try:
            from .settingscontroller import get_settings_sync
            from .cameraai import translate_raw_pose_to_pose_dict
            from .common import Frame
            import numpy as np

            settings = get_settings_sync()
            ai_setup = settings.get("aiSetup", {}) if isinstance(settings, dict) else {}

            pose_dict = None
            if raw_pose is not None:
                try:
                    pose_dict = translate_raw_pose_to_pose_dict(
                        raw_pose=raw_pose, ai_setup=ai_setup,
                    )
                except Exception as exc:
                    print(f"Error translating pose for {self._camera_name}: {exc}")
                    pose_dict = None

            # AIAgent.engage needs a Frame object so it can resolve the
            # reticle position from the image shape.  We don't have the
            # decoded AI image here (it lives in the worker), so we use a
            # small placeholder that carries the expected resolution.
            if image_width > 0 and image_height > 0:
                width, height = image_width, image_height
            else:
                cam_settings = self.settings
                width = int(cam_settings.get("stretch_width") or cam_settings.get("width") or 640)
                height = int(cam_settings.get("stretch_height") or cam_settings.get("height") or 480)
            placeholder = np.zeros((height, width, 3), dtype=np.uint8)
            frame = Frame(valid=True, image=placeholder, time=frame_time, pose=pose_dict)
            engagement_result = ai_agent.engage(frame=frame, settings=settings)
            snapshot = ai_agent.build_engagement_snapshot(
                engagement_result=engagement_result,
                settings=settings if isinstance(settings, dict) else {},
            )
            controller.send_engagement_result(
                result_payload=snapshot,
                pose_dict=pose_dict,
            )
        except Exception as exc:
            print(f"Error processing pose for {self._camera_name}: {exc}")

    def _should_stay_alive(self) -> bool:
        if self._is_headless_ai_processing_required():
            return True
        raw_t, masked_t, ai_t = self.get_last_access_times()
        most_recent = max(raw_t, masked_t, ai_t)
        return (time.time() - most_recent) <= self.TIMEOUT_SECONDS

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._invalidate_stream_tokens()
        self._active = False
        if threading.current_thread() is self:
            return
        if not self.is_alive():
            mc = self._master_controller
            controller = getattr(mc, "cameras_controller", None) if mc is not None else None
            if controller is not None:
                try:
                    controller.deactivate_camera(self)
                except Exception:
                    pass
            return
        self.join(timeout=self.STOP_TIMEOUT_SECONDS)
        if self.is_alive():
            print(
                f"Camera proxy still alive after stop timeout "
                f"(index={self._index}, camera_index={self._camera_index}, "
                f"camera_code={self._camera_code}, camera_name={self._camera_name})"
            )
