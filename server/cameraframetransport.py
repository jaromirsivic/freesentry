"""
Cross-process frame transport using shared memory.

SharedFrameSlot holds a single frame in a shared memory segment.
FrameTransport bundles three slots (raw, masked, ai) with pipes for
pose data and control commands.

All multiprocessing primitives are created through an explicit "spawn"
context so that behaviour is identical on Linux, macOS, and Windows.
"""

from __future__ import annotations

import atexit
import multiprocessing
import multiprocessing.shared_memory
import struct
import time
from typing import Any

import numpy as np

_mp_ctx = multiprocessing.get_context("spawn")

HEADER_FORMAT = "<IIIdBQQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

_MAX_SHM_NAME_LEN = 28


def _make_unique_id() -> str:
    import os
    raw = (os.getpid() ^ int(time.time_ns() & 0xFFFFFFFF))
    return format(raw & 0xFFFFFFFF, "08x")


class SharedFrameSlot:
    """One shared-memory region that holds a single video frame.

    The region layout is:
        [HEADER_SIZE bytes]  struct-packed header
        [data area]          raw pixel bytes (up to *max_data_size*)

    Header fields (little-endian):
        width          uint32
        height         uint32
        channels       uint32
        timestamp      float64
        valid          uint8 (bool)
        sequence       uint64
        data_size      uint64
    """

    def __init__(
        self,
        *,
        name: str,
        max_data_size: int,
        create: bool,
        lock: multiprocessing.synchronize.Lock,
    ) -> None:
        if len(name) > _MAX_SHM_NAME_LEN:
            raise ValueError(
                f"Shared memory name {name!r} exceeds {_MAX_SHM_NAME_LEN} chars"
            )
        self._name = name
        self._max_data_size = max_data_size
        self._lock = lock
        total_size = HEADER_SIZE + max_data_size
        if create:
            try:
                stale = multiprocessing.shared_memory.SharedMemory(
                    name=name, create=False
                )
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            self._shm = multiprocessing.shared_memory.SharedMemory(
                name=name, create=True, size=total_size
            )
        else:
            self._shm = multiprocessing.shared_memory.SharedMemory(
                name=name, create=False
            )

    @property
    def name(self) -> str:
        return self._name

    @property
    def lock(self) -> multiprocessing.synchronize.Lock:
        return self._lock

    def write_frame(
        self,
        image: np.ndarray,
        timestamp: float,
        valid: bool,
        sequence: int,
    ) -> None:
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim == 3 else 1
        data = image.tobytes()
        data_len = len(data)
        if data_len > self._max_data_size:
            raise ValueError(
                f"Frame data ({data_len} bytes) exceeds slot capacity "
                f"({self._max_data_size} bytes)"
            )
        header = struct.pack(
            HEADER_FORMAT, w, h, c, timestamp, int(valid), sequence, data_len
        )
        with self._lock:
            buf = self._shm.buf
            buf[:HEADER_SIZE] = header
            buf[HEADER_SIZE : HEADER_SIZE + data_len] = data

    def read_header(self) -> tuple[int, int, int, float, bool, int, int]:
        """Read just the header without copying pixel data.

        Returns (width, height, channels, timestamp, valid, sequence, data_size).
        """
        with self._lock:
            raw = bytes(self._shm.buf[:HEADER_SIZE])
        w, h, c, ts, v, seq, dsz = struct.unpack(HEADER_FORMAT, raw)
        return w, h, c, ts, bool(v), seq, dsz

    def read_frame(
        self,
    ) -> tuple[np.ndarray | None, float, bool, int]:
        """Copy the current frame out of shared memory.

        Returns (image_or_None, timestamp, valid, sequence).
        """
        with self._lock:
            raw_header = bytes(self._shm.buf[:HEADER_SIZE])
            w, h, c, ts, v, seq, dsz = struct.unpack(HEADER_FORMAT, raw_header)
            if dsz == 0 or not v:
                return None, ts, bool(v), seq
            pixel_bytes = bytearray(
                self._shm.buf[HEADER_SIZE : HEADER_SIZE + dsz]
            )
        shape = (h, w, c) if c > 1 else (h, w)
        image = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape(shape)
        return image, ts, True, seq

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
    """Bundles three frame slots with control / result pipes.

    Created by the **main process** (``create=True``).  The worker process
    reconstructs lightweight handles via ``connect_in_worker``.
    """

    SLOT_CHARS = {"raw": "r", "masked": "m", "ai": "a"}

    def __init__(
        self,
        *,
        camera_index: int,
        max_width: int = 1920,
        max_height: int = 1080,
    ) -> None:
        uid = _make_unique_id()
        self._max_data_size = max_width * max_height * 3
        self._slots: dict[str, SharedFrameSlot] = {}
        self._slot_locks: dict[str, multiprocessing.synchronize.Lock] = {}
        self._shm_names: dict[str, str] = {}
        for slot_key, char in self.SLOT_CHARS.items():
            name = f"fs_c{camera_index}{char}_{uid}"
            lock = _mp_ctx.Lock()
            self._slot_locks[slot_key] = lock
            self._shm_names[slot_key] = name
            self._slots[slot_key] = SharedFrameSlot(
                name=name,
                max_data_size=self._max_data_size,
                create=True,
                lock=lock,
            )

        self._command_parent, self._command_child = _mp_ctx.Pipe()
        self._pose_parent, self._pose_child = _mp_ctx.Pipe()

        self._cleaned_up = False
        atexit.register(self.cleanup)

    @property
    def slot_raw(self) -> SharedFrameSlot:
        return self._slots["raw"]

    @property
    def slot_masked(self) -> SharedFrameSlot:
        return self._slots["masked"]

    @property
    def slot_ai(self) -> SharedFrameSlot:
        return self._slots["ai"]

    @property
    def command_parent_conn(self) -> Any:
        return self._command_parent

    @property
    def command_child_conn(self) -> Any:
        return self._command_child

    @property
    def pose_parent_conn(self) -> Any:
        return self._pose_parent

    @property
    def pose_child_conn(self) -> Any:
        return self._pose_child

    def get_worker_init_args(self) -> dict[str, Any]:
        """Return pickle-safe arguments for ``CameraWorkerProcess.__init__``."""
        return {
            "shm_names": (
                self._shm_names["raw"],
                self._shm_names["masked"],
                self._shm_names["ai"],
            ),
            "shm_max_data_size": self._max_data_size,
            "slot_locks": (
                self._slot_locks["raw"],
                self._slot_locks["masked"],
                self._slot_locks["ai"],
            ),
            "pose_pipe_conn": self._pose_child,
            "command_pipe_conn": self._command_child,
        }

    def send_command(self, msg: dict[str, Any]) -> None:
        try:
            self._command_parent.send(msg)
        except (BrokenPipeError, OSError):
            pass

    def poll_pose(self, timeout: float = 0) -> bool:
        try:
            return self._pose_parent.poll(timeout)
        except (BrokenPipeError, OSError, EOFError):
            return False

    def recv_pose(self) -> Any:
        return self._pose_parent.recv()

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        for conn in (
            self._command_parent,
            self._command_child,
            self._pose_parent,
            self._pose_child,
        ):
            try:
                conn.close()
            except Exception:
                pass
        for slot in self._slots.values():
            slot.close()
            slot.unlink()
