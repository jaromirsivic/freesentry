"""End-to-end integration test for the Variant-4 ring reset.

Spawns a real :class:`server.cameraworker.CameraWorkerProcess` bound to
a dummy camera device and exercises the ``select_camera`` ->
``release_camera`` -> ``select_camera`` sequence to prove that
``_reset_rings_if_idle`` actually wipes the shared-memory ring buffers
at the start of the second ``_apply_select``.

The test is deterministic without needing any per-ms timing:

1. First session runs with ``demand_raw=True`` until a JPEG lands in
   the ``raw`` ring.
2. ``release_camera`` is sent; the ring is confirmed to still hold the
   stale JPEG (documents current behaviour on the stop path).
3. ``update_demand(raw=False)`` disables writes for the next session.
4. A second ``select_camera`` is sent for the same camera_index.  With
   demand disabled the worker never publishes a new frame, so the
   ``raw`` ring stays in whatever state ``_apply_select`` left it in.
5. The test asserts ``read_latest`` returns the empty sentinel
   (``data is None``, ``sequence == 0``, ``camera_index == -1``) --
   which can only happen if ``_reset_rings_if_idle`` cleared the ring.

The test is heavier than the primitives in ``test_ring_reset`` because
it actually spawns a subprocess that imports cv2 + numpy; it is
skipped automatically if those imports fail (e.g. minimal CI image).
"""

from __future__ import annotations

import threading
import time
import unittest

try:
    import cv2  # noqa: F401
    import numpy  # noqa: F401
    _DEPS_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on minimal runners only
    _DEPS_AVAILABLE = False

from server.cameraframetransport import FrameTransport


# Generous timeouts: Windows "spawn" + cv2 + numpy import can take
# several seconds on cold caches.
_WORKER_BOOT_TIMEOUT = 15.0
_STATE_TRANSITION_TIMEOUT = 8.0
_FIRST_FRAME_TIMEOUT = 10.0


class _WorkerHarness:
    """Spawns a :class:`CameraWorkerProcess`, relays heartbeats, and
    exposes the latest worker status snapshot.

    The harness runs a background thread that drains the pose pipe (a
    mix of ``{"type": "heartbeat", ...}`` and ``{"type": "pose", ...}``
    messages) and also periodically sends ``heartbeat`` commands back
    to the worker so its idle-timeout does not fire mid-test.
    """

    def __init__(self) -> None:
        # Importing inside __init__ so the class is still definable
        # on runners where cv2/numpy are missing (the test itself is
        # skipped separately).
        from server.cameraworker import CameraWorkerProcess

        self.transport = FrameTransport(max_jpeg_size=512 * 1024)
        init_args = self.transport.get_worker_init_args()
        self.worker = CameraWorkerProcess(**init_args)

        self._stop_event = threading.Event()
        self._latest_status_lock = threading.Lock()
        self._latest_status: dict | None = None
        self._latest_status_mono: float = 0.0

        self._pose_thread = threading.Thread(
            target=self._drain_pose_loop,
            name="test-harness-pose-drain",
            daemon=True,
        )
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="test-harness-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self.worker.start()
        self._pose_thread.start()
        self._heartbeat_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Ask worker to exit.
        try:
            self.transport.send_command({"cmd": "stop"})
        except Exception:
            pass
        if self.worker.is_alive():
            self.worker.join(timeout=5.0)
        if self.worker.is_alive():
            try:
                self.worker.terminate()
            except Exception:
                pass
            self.worker.join(timeout=2.0)
        # Drain threads.
        self._pose_thread.join(timeout=1.0)
        self._heartbeat_thread.join(timeout=1.0)
        try:
            self.transport.cleanup()
        except Exception:
            pass

    # -- background relays ---------------------------------------------

    def _drain_pose_loop(self) -> None:
        """Consume everything the worker pushes on its pose pipe so the
        pipe's OS-level buffer never fills up and blocks the worker's
        heartbeat thread.  Cache the most recent status snapshot for
        the test-body assertions.
        """
        conn = self.transport.pose_parent_conn
        while not self._stop_event.is_set():
            try:
                if not conn.poll(0.1):
                    continue
                msg = conn.recv()
            except (EOFError, OSError, BrokenPipeError):
                return
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("type") == "heartbeat":
                status = msg.get("status")
                if isinstance(status, dict):
                    with self._latest_status_lock:
                        self._latest_status = status
                        self._latest_status_mono = time.monotonic()

    def _heartbeat_loop(self) -> None:
        """Send heartbeat commands to the worker at ~4 Hz so its
        internal ``HEARTBEAT_TIMEOUT`` never trips inside the test.
        """
        while not self._stop_event.is_set():
            try:
                self.transport.send_command({"cmd": "heartbeat"})
            except Exception:
                return
            if self._stop_event.wait(timeout=0.25):
                return

    # -- status helpers ------------------------------------------------

    def get_status(self) -> tuple[dict | None, float]:
        with self._latest_status_lock:
            return self._latest_status, self._latest_status_mono

    def wait_for_status(
        self,
        *,
        predicate,
        timeout: float,
        description: str,
    ) -> dict:
        """Block until ``predicate(status)`` returns True.  Returns
        the matching status snapshot; raises ``AssertionError`` on
        timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._latest_status_lock:
                status = self._latest_status
            if status is not None and predicate(status):
                return status
            time.sleep(0.05)
        raise AssertionError(
            f"Timed out waiting for worker status: {description} "
            f"(last status seen: {self._latest_status})"
        )


@unittest.skipUnless(
    _DEPS_AVAILABLE,
    "cv2/numpy not importable; skipping worker integration test",
)
class WorkerRingResetIntegrationTests(unittest.TestCase):
    """Full round-trip: select -> release -> select clears the rings."""

    def setUp(self) -> None:
        self.harness = _WorkerHarness()
        self.harness.start()
        # Wait for the worker's first heartbeat so we know the process
        # is alive, the heartbeat thread is up, and ``_latest_status``
        # is populated.
        self.harness.wait_for_status(
            predicate=lambda _s: True,
            timeout=_WORKER_BOOT_TIMEOUT,
            description="first heartbeat after worker spawn",
        )

    def tearDown(self) -> None:
        try:
            self.harness.stop()
        except Exception:
            pass

    def _send_select(self, *, camera_index: int) -> None:
        self.harness.transport.send_command({
            "cmd": "select_camera",
            "camera_type": "dummy",
            "camera_index": camera_index,
            "index": camera_index,
            "camera_code": "scope_camera",
            "camera_name": f"test_dummy_{camera_index}",
            "camera_settings": {},
            "global_settings": {},
        })

    def _send_demand(self, *, raw: bool, masked: bool = False, ai: bool = False) -> None:
        self.harness.transport.send_command({
            "cmd": "update_demand",
            "raw": raw,
            "masked": masked,
            "ai": ai,
            "jpeg_quality_raw": 50,
            "jpeg_quality_masked": 50,
            "jpeg_quality_ai": 50,
        })

    def _send_release(self) -> None:
        self.harness.transport.send_command({"cmd": "release_camera"})

    def _wait_for_frame(self, *, mode_key: str, timeout: float) -> tuple[bytes, int, int]:
        """Spin-read the ring until a frame is available.  Returns
        ``(data, camera_index, sequence)``.
        """
        ring = self.harness.transport.ring(mode_key)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data, _ts, _mode, cam, seq = ring.read_latest(since_sequence=0)
            if data is not None and seq > 0:
                return data, cam, seq
            time.sleep(0.05)
        raise AssertionError(
            f"Timed out waiting for first frame in ring '{mode_key}'"
        )

    def test_select_after_release_clears_ring(self) -> None:
        # --- session 1: publish at least one frame ------------------
        # Order matters: demand first, then select, so the worker
        # starts publishing as soon as the device opens.
        self._send_demand(raw=True)
        self._send_select(camera_index=0)

        self.harness.wait_for_status(
            predicate=lambda s: s.get("device_state") == "open"
            and s.get("active_camera_index") == 0,
            timeout=_STATE_TRANSITION_TIMEOUT,
            description="device_state=open after first select",
        )

        stale_data, stale_cam, stale_seq = self._wait_for_frame(
            mode_key="raw", timeout=_FIRST_FRAME_TIMEOUT,
        )
        self.assertIsNotNone(stale_data)
        self.assertEqual(stale_cam, 0)
        self.assertGreaterEqual(stale_seq, 1)

        # --- release: ring must still carry the stale frame ---------
        self._send_release()
        self.harness.wait_for_status(
            predicate=lambda s: s.get("device_state") == "closed"
            and s.get("active_camera_index") == -1,
            timeout=_STATE_TRANSITION_TIMEOUT,
            description="device_state=closed after release",
        )
        ring = self.harness.transport.ring("raw")
        data, _ts, _mode, cam, seq = ring.read_latest(since_sequence=0)
        self.assertIsNotNone(
            data,
            "release_camera must NOT clear the ring (stop path stays cheap)",
        )
        self.assertEqual(cam, 0)
        self.assertEqual(seq, stale_seq)

        # --- session 2: disable demand, re-select same camera -------
        # With demand_raw=False, the worker will run _apply_select
        # (which calls _reset_rings_if_idle) but will NOT publish any
        # new frame.  The only way the ring can end up empty is via
        # the reset.
        self._send_demand(raw=False)
        self._send_select(camera_index=0)

        # Wait for select to have been applied on the worker side.
        # We detect this by watching for a status snapshot where the
        # device is open again AND ``select_applied_mono`` has moved
        # to a value newer than the stale frame's local-clock timestamp.
        #
        # In practice ``device_state == "open"`` after a *new*
        # select is a sufficient marker because _apply_select calls
        # _close_device (device_state=closed) before reopening.
        t_select_sent = time.monotonic()
        self.harness.wait_for_status(
            predicate=lambda s: (
                s.get("device_state") == "open"
                and s.get("active_camera_index") == 0
                and float(s.get("select_applied_mono", 0.0)) >= t_select_sent
            ),
            timeout=_STATE_TRANSITION_TIMEOUT,
            description="fresh device_state=open after second select",
        )

        # Give the worker a small grace window to run a few main-loop
        # iterations in the demand-disabled state.  Because demand is
        # False, these iterations only idle-sleep; they cannot
        # re-publish a frame that would mask the reset.
        time.sleep(0.3)

        data, _ts, _mode, cam, seq = ring.read_latest(since_sequence=0)
        self.assertIsNone(
            data,
            "ring must be empty after second _apply_select; "
            "_reset_rings_if_idle did not wipe it",
        )
        self.assertEqual(seq, 0)
        self.assertEqual(cam, -1)


if __name__ == "__main__":
    unittest.main()
