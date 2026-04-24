"""Tests for the JPEG ring buffer reset primitive and its guard.

Covers:

* :meth:`server.cameraframetransport.JpegRingBuffer.reset` primitive --
  after a ``write`` + ``reset`` cycle, ``read_latest`` returns the
  empty sentinel and the *next* ``write`` is visible as the fresh
  ``latest_idx``.  Also that ``wait_for_new`` no longer returns True
  for events ``set`` before the reset.

* :func:`server.cameraworker._should_reset_rings` predicate (the
  testable extract of the guard used by
  ``_reset_rings_if_idle`` inside :class:`CameraWorkerProcess.run`).

* End-to-end guard harness: a closure that mirrors
  ``_reset_rings_if_idle`` runs ``reset()`` only when the predicate
  passes, so we can directly assert "the ring still holds the
  previous frame when the worker is 'busy'" without spawning a
  subprocess.

No worker subprocess is spawned here -- we build a single
``JpegRingBuffer`` with ``create=True`` and exercise both sides
from the test process.  Spawn-context locks are still used so the
cross-process-lock contract is faithful.
"""

from __future__ import annotations

import multiprocessing
import os
import time
import unittest

from server.cameraframetransport import (
    DEFAULT_MAX_JPEG_SIZE,
    JpegRingBuffer,
    RING_SLOTS,
)
from server.cameraworker import _should_reset_rings


_mp_ctx = multiprocessing.get_context("spawn")


def _make_ring(name: str, *, max_jpeg_size: int = 64 * 1024) -> JpegRingBuffer:
    """Build a self-contained :class:`JpegRingBuffer` for unit tests.

    Keeps the slot size tiny (64 KiB) so the shared-memory segment
    created here is ~192 KiB instead of the production 6 MiB.
    """
    slot_locks = (_mp_ctx.Lock(), _mp_ctx.Lock(), _mp_ctx.Lock())
    index_lock = _mp_ctx.Lock()
    new_frame_event = _mp_ctx.Event()
    return JpegRingBuffer(
        name=name,
        max_jpeg_size=max_jpeg_size,
        create=True,
        slot_locks=slot_locks,
        index_lock=index_lock,
        new_frame_event=new_frame_event,
    )


def _unique_name(tag: str) -> str:
    """Short, unique shared-memory segment name.

    ``JpegRingBuffer`` enforces a <=18 char limit so we must build a
    name that fits even on macOS/POSIX.  ``tag`` is truncated as
    needed.
    """
    # up to 18 chars: "t_" (2) + pid%10000 (<=4) + "_" (1) + ns%100000 (<=5) + "_" (1) + tag (<=5)
    pid = os.getpid() % 10000
    ns = time.time_ns() % 100000
    short_tag = tag[:5]
    return f"t_{pid}_{ns}_{short_tag}"


class JpegRingBufferResetPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.name = _unique_name("reset")
        self.ring = _make_ring(self.name)

    def tearDown(self) -> None:
        try:
            self.ring.close()
        except Exception:
            pass
        try:
            self.ring.unlink()
        except Exception:
            pass

    def test_write_is_visible_before_reset(self) -> None:
        payload = b"\xff\xd8hello\xff\xd9"
        self.ring.write(
            data=payload, timestamp=123.456, mode=0, camera_index=2, sequence=7,
        )
        data, ts, mode, cam, seq = self.ring.read_latest(since_sequence=0)
        self.assertEqual(data, payload)
        self.assertAlmostEqual(ts, 123.456)
        self.assertEqual(mode, 0)
        self.assertEqual(cam, 2)
        self.assertEqual(seq, 7)

    def test_reset_empties_ring(self) -> None:
        self.ring.write(
            data=b"payload", timestamp=1.0, mode=0, camera_index=2, sequence=42,
        )
        self.ring.reset()
        data, ts, mode, cam, seq = self.ring.read_latest(since_sequence=0)
        self.assertIsNone(data)
        # Sentinel for "no frame" -- caller distinguishes it from
        # "not newer than since" via sequence == 0.
        self.assertEqual(seq, 0)
        self.assertEqual(cam, -1)
        self.assertEqual(mode, 0)
        self.assertEqual(ts, 0.0)

    def test_write_after_reset_lands_at_sequence_one(self) -> None:
        """The producer ABI is that seq counters restart at 0 after a
        ``_apply_select`` (and therefore after a ring reset), so the
        very next ``write`` carries ``sequence=1``.  That first post-
        reset write must be readable immediately and must NOT resurrect
        the pre-reset frame.
        """
        self.ring.write(
            data=b"stale", timestamp=10.0, mode=0, camera_index=2, sequence=5,
        )
        self.ring.reset()
        # Empty now.
        data, _, _, _, seq = self.ring.read_latest(since_sequence=0)
        self.assertIsNone(data)
        self.assertEqual(seq, 0)

        fresh = b"fresh-frame"
        self.ring.write(
            data=fresh, timestamp=20.0, mode=0, camera_index=2, sequence=1,
        )
        data, ts, mode, cam, seq = self.ring.read_latest(since_sequence=0)
        self.assertEqual(data, fresh)
        self.assertEqual(seq, 1)
        self.assertEqual(cam, 2)
        self.assertEqual(mode, 0)
        self.assertAlmostEqual(ts, 20.0)

    def test_wait_for_new_returns_false_immediately_after_reset(self) -> None:
        """``write`` calls ``new_frame_event.set()``.  Without an
        intervening consumer, the event remains set and
        ``wait_for_new`` would return True spuriously for the next
        session.  ``reset`` must clear the event so a consumer that
        just started waiting sees "no new frame yet" and blocks for
        the full timeout.
        """
        self.ring.write(
            data=b"abc", timestamp=1.0, mode=0, camera_index=0, sequence=1,
        )
        self.ring.reset()
        start = time.monotonic()
        woken = self.ring.wait_for_new(timeout=0.05)
        elapsed = time.monotonic() - start
        self.assertFalse(woken)
        # Must have actually waited close to the full timeout, not
        # returned immediately via the lingering set() signal.
        self.assertGreaterEqual(elapsed, 0.04)

    def test_reset_wipes_all_slots_not_just_latest(self) -> None:
        """Producer writes to ``(latest_idx + 1) % 3`` each time, so
        three consecutive writes visit every slot.  After the reset,
        ``latest_idx`` must be back to 0 and every slot header must
        read empty regardless of which one ``latest_idx`` points at.
        """
        for i in range(RING_SLOTS + 1):
            self.ring.write(
                data=f"frame-{i}".encode(), timestamp=float(i),
                mode=0, camera_index=0, sequence=i + 1,
            )
        self.ring.reset()
        data, _, _, _, seq = self.ring.read_latest(since_sequence=0)
        self.assertIsNone(data)
        self.assertEqual(seq, 0)

        # Next write goes to slot (0 + 1) % 3 = 1, which definitely
        # contained a stale frame before reset (sequence 2 or 5).
        self.ring.write(
            data=b"after-reset", timestamp=99.0,
            mode=0, camera_index=0, sequence=1,
        )
        data, _, _, _, seq = self.ring.read_latest(since_sequence=0)
        self.assertEqual(data, b"after-reset")
        self.assertEqual(seq, 1)


class ShouldResetRingsPredicateTests(unittest.TestCase):
    """Unit tests for the guard predicate used by
    ``_reset_rings_if_idle`` in :class:`CameraWorkerProcess.run`.
    """

    def test_idle_worker_allows_reset(self) -> None:
        self.assertTrue(_should_reset_rings(
            device=None, ring_write_in_progress=False,
        ))

    def test_device_still_open_blocks_reset(self) -> None:
        self.assertFalse(_should_reset_rings(
            device=object(), ring_write_in_progress=False,
        ))

    def test_ring_write_in_flight_blocks_reset(self) -> None:
        self.assertFalse(_should_reset_rings(
            device=None, ring_write_in_progress=True,
        ))

    def test_both_conditions_bad_blocks_reset(self) -> None:
        self.assertFalse(_should_reset_rings(
            device=object(), ring_write_in_progress=True,
        ))


class GuardedResetHarnessTests(unittest.TestCase):
    """Integration-flavoured test: drive a mini-harness that mirrors
    :func:`server.cameraworker._reset_rings_if_idle` so we can assert
    the end-to-end behavior (including "ring still holds previous
    frame when worker is busy") without spawning a real subprocess.
    """

    def setUp(self) -> None:
        self.name = _unique_name("guard")
        self.ring = _make_ring(self.name)

    def tearDown(self) -> None:
        try:
            self.ring.close()
        except Exception:
            pass
        try:
            self.ring.unlink()
        except Exception:
            pass

    @staticmethod
    def _reset_rings_if_idle(
        *, device, ring_write_in_progress: bool, rings,
    ) -> bool:
        """Mirrors the production helper in cameraworker.py.  Returns
        True iff the reset actually ran.
        """
        if not _should_reset_rings(
            device=device, ring_write_in_progress=ring_write_in_progress,
        ):
            return False
        for ring in rings.values():
            ring.reset()
        return True

    def test_guard_skips_reset_when_write_in_progress(self) -> None:
        self.ring.write(
            data=b"stale-session", timestamp=1.0,
            mode=0, camera_index=2, sequence=5,
        )
        did_reset = self._reset_rings_if_idle(
            device=None,
            ring_write_in_progress=True,
            rings={"raw": self.ring},
        )
        self.assertFalse(did_reset)

        # Ring must still serve the pre-guard frame unchanged --
        # the whole point of "skip when busy" is that we accept the
        # status quo instead of racing the writer.
        data, _, _, cam, seq = self.ring.read_latest(since_sequence=0)
        self.assertEqual(data, b"stale-session")
        self.assertEqual(seq, 5)
        self.assertEqual(cam, 2)

    def test_guard_skips_reset_when_device_not_none(self) -> None:
        self.ring.write(
            data=b"stale-session", timestamp=1.0,
            mode=0, camera_index=2, sequence=5,
        )
        did_reset = self._reset_rings_if_idle(
            device=object(),
            ring_write_in_progress=False,
            rings={"raw": self.ring},
        )
        self.assertFalse(did_reset)

        data, _, _, _, seq = self.ring.read_latest(since_sequence=0)
        self.assertEqual(data, b"stale-session")
        self.assertEqual(seq, 5)

    def test_guard_runs_reset_when_idle(self) -> None:
        self.ring.write(
            data=b"stale-session", timestamp=1.0,
            mode=0, camera_index=2, sequence=5,
        )
        did_reset = self._reset_rings_if_idle(
            device=None,
            ring_write_in_progress=False,
            rings={"raw": self.ring},
        )
        self.assertTrue(did_reset)

        data, _, _, _, seq = self.ring.read_latest(since_sequence=0)
        self.assertIsNone(data)
        self.assertEqual(seq, 0)

    def test_guard_runs_reset_across_all_three_rings(self) -> None:
        """``_reset_rings_if_idle`` iterates ``rings.values()`` so all
        three JPEG modes (raw/masked/ai) get wiped in one call.  Mirror
        that with three independent rings to catch any bug where only
        the first is reset.
        """
        names = [_unique_name(f"r{i}") for i in range(3)]
        rings = {
            f"ring{i}": _make_ring(n) for i, n in enumerate(names)
        }
        try:
            for i, ring in enumerate(rings.values()):
                ring.write(
                    data=f"stale-{i}".encode(),
                    timestamp=float(i),
                    mode=0,
                    camera_index=i,
                    sequence=i + 10,
                )
            # All three rings currently have frames visible.
            for ring in rings.values():
                data, _, _, _, seq = ring.read_latest(since_sequence=0)
                self.assertIsNotNone(data)
                self.assertGreater(seq, 0)

            did_reset = self._reset_rings_if_idle(
                device=None,
                ring_write_in_progress=False,
                rings=rings,
            )
            self.assertTrue(did_reset)

            # All three must now be empty.
            for ring in rings.values():
                data, _, _, _, seq = ring.read_latest(since_sequence=0)
                self.assertIsNone(data)
                self.assertEqual(seq, 0)
        finally:
            for ring in rings.values():
                try:
                    ring.close()
                except Exception:
                    pass
                try:
                    ring.unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
