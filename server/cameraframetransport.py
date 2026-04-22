"""
Cross-process JPEG frame transport using shared memory.

A single :class:`FrameTransport` is created by the main process and owns the
sole :class:`CameraWorkerProcess`.  The worker JPEG-encodes every outgoing
frame (raw, masked, AI) and writes the bytes into a :class:`JpegRingBuffer`
(triple-buffered per mode) so that the main process never has to re-encode
and the HTTP relay can simply copy bytes out.

All multiprocessing primitives are created through an explicit ``spawn``
context so behaviour is identical on Windows, Linux, and Raspberry Pi.

Triple-buffer protocol (per mode)
---------------------------------
Producer (camera worker):

1. Reads ``latest_idx`` from the footer.
2. Picks ``target = (latest_idx + 1) % 3`` — a slot that is neither "just
   published" nor (in steady state) "currently being read".
3. Takes that slot's per-slot lock and writes ``header + jpeg_bytes``.
4. Atomically updates ``latest_idx`` to ``target`` under ``index_lock``.
5. Sets the per-ring ``new_frame_event`` so any consumer waiting on it
   wakes up immediately.

Consumer (main process):

1. Calls ``read_latest(since_sequence=...)``.  The read is **lossy
   newest-wins**: it returns whatever ``latest_idx`` points to; any frames
   published between the previous read and this one are silently dropped.
   This is exactly what we want — if the main relay lags behind the camera
   worker, we jump straight to the freshest frame instead of building a
   queue.
2. If no new frame is available it may briefly ``wait_for_new(timeout)`` on
   the per-ring event to avoid spinning while still returning in bounded
   time.
3. Once a slot is read, the consumer does **not** need to ack.  With three
   slots and the ``(latest + 1) % 3`` producer rule, there is always at
   least one "spare" slot between the writer and the reader, and the
   per-slot lock prevents the rare case where the reader happens to be
   decoding the slot that becomes the next writer target.  An explicit ack
   would only add latency for zero correctness benefit.

Pipes
-----
* ``command_pipe``     : main  → worker  (select_camera, release_camera,
  update_demand, update_camera_settings, update_global_settings,
  update_ai_setup, heartbeat, stop)
* ``pose_pipe``        : worker → main   (pose samples, worker heartbeat)
* ``ai_result_pipe``   : main  → worker  (engagement result to overlay,
  plus the translated pose_dict and ai_setup used for drawing)
"""

from __future__ import annotations

import atexit
import multiprocessing
import multiprocessing.shared_memory
import os
import struct
import threading
import time
from typing import Any

_mp_ctx = multiprocessing.get_context("spawn")

# Header: sequence(u64) + timestamp(f64) + mode(u32) + camera_index(i32) + data_size(u32)
JPEG_HEADER_FORMAT = "<QdiiI"
JPEG_HEADER_SIZE = struct.calcsize(JPEG_HEADER_FORMAT)

# Shared-memory name length budget: keep well under the POSIX 30-char limit
# and the macOS 31-char limit.  Multiplatform checklist requires <=18.
_MAX_SHM_NAME_LEN = 18

# One JPEG slot cap.  2 MiB is plenty for 1080p at quality 90.
DEFAULT_MAX_JPEG_SIZE = 2 * 1024 * 1024

# Number of slots per mode (triple buffer).
RING_SLOTS = 3


def _make_unique_id() -> str:
    raw = (os.getpid() ^ int(time.time_ns() & 0xFFFFFFFF))
    return format(raw & 0xFFFFFFFF, "08x")


class JpegRingBuffer:
    """Triple-buffered JPEG slot in shared memory.

    Layout::

        [ slot_0 ][ slot_1 ][ slot_2 ][ latest_idx (u32) ]

    Each slot holds ``JPEG_HEADER_SIZE + max_jpeg_size`` bytes.  The last
    4 bytes of the segment hold the index of the most recently written
    valid slot (producer published).

    The producer (worker) always writes to ``(latest_idx + 1) % RING_SLOTS``
    which is guaranteed to be neither the currently published slot nor the
    slot the consumer is presumably reading (with a single main-process
    consumer and the 3-slot rotation), giving a lock-held window small
    enough to be effectively non-blocking.

    ``new_frame_event`` is set by the producer after publishing.  Consumers
    can ``wait_for_new(timeout)`` on it to sleep briefly until the next
    frame lands, avoiding CPU-busy polling while still providing bounded
    latency.  The read path itself is still lossy newest-wins: if several
    frames were published while the consumer was away, only the most
    recent is returned.
    """

    def __init__(
        self,
        *,
        name: str,
        max_jpeg_size: int,
        create: bool,
        slot_locks: tuple[Any, Any, Any],
        index_lock: Any,
        new_frame_event: Any,
    ) -> None:
        if len(name) > _MAX_SHM_NAME_LEN:
            raise ValueError(
                f"Shared memory name {name!r} exceeds {_MAX_SHM_NAME_LEN} chars"
            )
        self._name = name
        self._max_jpeg_size = max_jpeg_size
        self._slot_size = JPEG_HEADER_SIZE + max_jpeg_size
        self._total_size = self._slot_size * RING_SLOTS + 4
        self._slot_locks = slot_locks
        self._index_lock = index_lock
        self._new_frame_event = new_frame_event

        if create:
            try:
                stale = multiprocessing.shared_memory.SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            self._shm = multiprocessing.shared_memory.SharedMemory(
                name=name, create=True, size=self._total_size
            )
            # Explicitly clear latest_idx (0 is valid but slot 0 has seq=0 so
            # readers treat it as "no frame yet").
            self._shm.buf[self._total_size - 4:self._total_size] = struct.pack("<I", 0)
        else:
            self._shm = multiprocessing.shared_memory.SharedMemory(name=name, create=False)

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_jpeg_size(self) -> int:
        return self._max_jpeg_size

    # ------------------------------------------------------------------
    # Writer / reader primitives
    # ------------------------------------------------------------------
    def _get_latest_idx(self) -> int:
        with self._index_lock:
            raw = bytes(self._shm.buf[self._total_size - 4:self._total_size])
        return struct.unpack("<I", raw)[0] & 0x03  # clamp to 0..3

    def _set_latest_idx(self, idx: int) -> None:
        with self._index_lock:
            self._shm.buf[self._total_size - 4:self._total_size] = struct.pack("<I", idx)

    def write(
        self,
        *,
        data: bytes,
        timestamp: float,
        mode: int,
        camera_index: int,
        sequence: int,
    ) -> None:
        data_len = len(data)
        if data_len > self._max_jpeg_size:
            raise ValueError(
                f"JPEG data ({data_len} bytes) exceeds slot capacity "
                f"({self._max_jpeg_size} bytes)"
            )
        latest = self._get_latest_idx()
        target = (latest + 1) % RING_SLOTS
        header = struct.pack(
            JPEG_HEADER_FORMAT, sequence, timestamp, mode, camera_index, data_len,
        )
        offset = target * self._slot_size
        with self._slot_locks[target]:
            buf = self._shm.buf
            buf[offset:offset + JPEG_HEADER_SIZE] = header
            buf[offset + JPEG_HEADER_SIZE:offset + JPEG_HEADER_SIZE + data_len] = data
        self._set_latest_idx(target)
        # Wake any consumer waiting for a new frame.  A spurious wake is
        # harmless: the consumer re-checks sequence numbers before yielding.
        try:
            self._new_frame_event.set()
        except Exception:
            pass

    def read_latest(
        self,
        *,
        since_sequence: int = 0,
    ) -> tuple[bytes | None, float, int, int, int]:
        """Return ``(data, timestamp, mode, camera_index, sequence)``.

        Returns ``(None, 0.0, 0, -1, 0)`` when the buffer is empty (no
        frame has ever been published for this mode).  Returns
        ``(None, ts, mode, camera_index, sequence)`` with ``sequence > 0``
        when a frame exists but is not newer than *since_sequence* — i.e.
        the consumer already has it.  The caller can tell the two cases
        apart by inspecting ``sequence``.

        This call is **lossy newest-wins**: if several frames were
        published since the previous call, only the most recent is
        returned and the older ones are silently skipped.  This is the
        correct back-pressure behaviour for a live video stream — we
        never want to build up a queue of stale JPEGs.
        """
        idx = self._get_latest_idx() % RING_SLOTS
        offset = idx * self._slot_size
        with self._slot_locks[idx]:
            raw_header = bytes(self._shm.buf[offset:offset + JPEG_HEADER_SIZE])
            sequence, timestamp, mode, camera_index, data_len = struct.unpack(
                JPEG_HEADER_FORMAT, raw_header,
            )
            if data_len == 0 or sequence == 0 or sequence <= since_sequence:
                return None, timestamp, mode, camera_index, sequence
            data = bytes(
                self._shm.buf[offset + JPEG_HEADER_SIZE:offset + JPEG_HEADER_SIZE + data_len]
            )
        return data, timestamp, mode, camera_index, sequence

    def wait_for_new(self, timeout: float) -> bool:
        """Block until the producer publishes a new frame, or *timeout*
        elapses.  Returns ``True`` if woken by the event, ``False`` on
        timeout.

        The event is cleared before returning so the next call blocks
        again until the producer publishes once more.  This is a best-
        effort latency helper; spurious wakes are fine because callers
        re-check ``sequence`` via :meth:`read_latest` anyway.
        """
        try:
            woken = self._new_frame_event.wait(timeout=timeout)
            if woken:
                self._new_frame_event.clear()
            return bool(woken)
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        try:
            self._shm.unlink()
        except Exception:
            pass


class FrameTransport:
    """Singleton IPC transport shared by the main process and the one
    camera worker process.

    There is exactly one ``FrameTransport`` instance per application (owned
    by :class:`~server.camerascontroller.CamerasController`).  The worker
    dynamically switches between different physical cameras via
    ``select_camera`` / ``release_camera`` commands — **the transport
    itself never needs to be recreated when the user picks a different
    camera**.
    """

    MODES = ("raw", "masked", "ai")
    MODE_NUM = {"raw": 0, "masked": 1, "ai": 3}
    SLOT_CHARS = {"raw": "r", "masked": "m", "ai": "a"}

    def __init__(self, *, max_jpeg_size: int = DEFAULT_MAX_JPEG_SIZE) -> None:
        self._max_jpeg_size = max_jpeg_size
        uid = _make_unique_id()
        self._uid = uid

        self._rings: dict[str, JpegRingBuffer] = {}
        self._index_locks: dict[str, Any] = {}
        self._slot_lock_groups: dict[str, tuple[Any, Any, Any]] = {}
        self._new_frame_events: dict[str, Any] = {}
        self._shm_names: dict[str, str] = {}

        for mode_key, char in self.SLOT_CHARS.items():
            name = f"fj_{uid}_{char}"  # 3 + 8 + 1 + 1 = 13 chars
            slot_locks = (_mp_ctx.Lock(), _mp_ctx.Lock(), _mp_ctx.Lock())
            index_lock = _mp_ctx.Lock()
            new_frame_event = _mp_ctx.Event()
            self._slot_lock_groups[mode_key] = slot_locks
            self._index_locks[mode_key] = index_lock
            self._new_frame_events[mode_key] = new_frame_event
            self._shm_names[mode_key] = name
            self._rings[mode_key] = JpegRingBuffer(
                name=name,
                max_jpeg_size=max_jpeg_size,
                create=True,
                slot_locks=slot_locks,
                index_lock=index_lock,
                new_frame_event=new_frame_event,
            )

        # main -> worker
        self._command_parent, self._command_child = _mp_ctx.Pipe()
        # worker -> main (pose + heartbeat)
        self._pose_parent, self._pose_child = _mp_ctx.Pipe()
        # main -> worker (engagement results drawn by worker)
        self._ai_result_parent, self._ai_result_child = _mp_ctx.Pipe()

        # Locks to serialize concurrent sends from multiple threads
        # (HTTP handlers, camera proxies, controller IO thread).
        self._command_send_lock = threading.Lock()
        self._ai_result_send_lock = threading.Lock()

        self._cleaned_up = False
        atexit.register(self.cleanup)

    # ------------------------------------------------------------------
    # Accessors used by main process
    # ------------------------------------------------------------------
    @property
    def max_jpeg_size(self) -> int:
        return self._max_jpeg_size

    def ring(self, mode_key: str) -> JpegRingBuffer:
        return self._rings[mode_key]

    @property
    def command_parent_conn(self) -> Any:
        return self._command_parent

    @property
    def pose_parent_conn(self) -> Any:
        return self._pose_parent

    @property
    def ai_result_parent_conn(self) -> Any:
        return self._ai_result_parent

    # ------------------------------------------------------------------
    # Arguments for CameraWorkerProcess
    # ------------------------------------------------------------------
    def get_worker_init_args(self) -> dict[str, Any]:
        """Return picklable arguments that the worker uses to reconstruct
        lightweight handles via :meth:`JpegRingBuffer` with ``create=False``.
        """
        return {
            "shm_names": {k: self._shm_names[k] for k in self.MODES},
            "max_jpeg_size": self._max_jpeg_size,
            "slot_lock_groups": {k: self._slot_lock_groups[k] for k in self.MODES},
            "index_locks": {k: self._index_locks[k] for k in self.MODES},
            "new_frame_events": {k: self._new_frame_events[k] for k in self.MODES},
            "command_pipe_conn": self._command_child,
            "pose_pipe_conn": self._pose_child,
            "ai_result_pipe_conn": self._ai_result_child,
        }

    # ------------------------------------------------------------------
    # Command / data helpers
    # ------------------------------------------------------------------
    def send_command(self, msg: dict[str, Any]) -> None:
        try:
            with self._command_send_lock:
                self._command_parent.send(msg)
        except (BrokenPipeError, OSError, EOFError):
            pass

    def send_ai_result(self, msg: dict[str, Any]) -> None:
        try:
            with self._ai_result_send_lock:
                self._ai_result_parent.send(msg)
        except (BrokenPipeError, OSError, EOFError):
            pass

    def poll_pose(self, timeout: float = 0) -> bool:
        try:
            return self._pose_parent.poll(timeout)
        except (BrokenPipeError, OSError, EOFError):
            return False

    def recv_pose(self) -> Any:
        return self._pose_parent.recv()

    def read_jpeg(
        self, *, mode_key: str, since_sequence: int = 0,
    ) -> tuple[bytes | None, float, int, int, int]:
        return self._rings[mode_key].read_latest(since_sequence=since_sequence)

    def wait_for_new_jpeg(self, *, mode_key: str, timeout: float) -> bool:
        """Block up to *timeout* seconds waiting for the worker to publish
        a new frame in *mode_key*.  Returns ``True`` if woken by the
        producer, ``False`` on timeout.
        """
        return self._rings[mode_key].wait_for_new(timeout=timeout)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        for conn in (
            self._command_parent, self._command_child,
            self._pose_parent, self._pose_child,
            self._ai_result_parent, self._ai_result_child,
        ):
            try:
                conn.close()
            except Exception:
                pass
        for ring in self._rings.values():
            ring.close()
            ring.unlink()
