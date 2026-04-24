"""Focused tests for the MJPEG stall watchdog classifier.

Only exercises the pure-Python helper
``_watchdog_worker_legitimately_not_ready`` from
:mod:`server.restapicameras`.  No multiprocessing or HTTP machinery is
involved.  The helper is what decides whether the stall watchdog
should take recovery action or patiently wait for the worker to finish
something it is legitimately busy with (device opening, YOLO loading,
capture failing, select_camera still propagating).
"""

from __future__ import annotations

import time
import unittest

from server.restapicameras import _watchdog_worker_legitimately_not_ready


def _ready_status(*, camera_index: int = 0) -> dict:
    """Baseline 'worker is fully up and serving this camera' snapshot.

    Individual tests start from this and mutate the one field they
    want to probe.
    """
    now_mono = time.monotonic()
    return {
        "active_camera_index": camera_index,
        "active_camera_code": "scope_camera",
        "select_applied_mono": now_mono - 5.0,
        "device_state": "open",
        "last_capture_ok_mono": now_mono - 0.05,
        "last_capture_fail_mono": 0.0,
        "ai_inference_started_mono": now_mono - 0.1,
        "ai_inference_completed_mono": now_mono - 0.08,
        "ai_model_state": "ready",
        "demand_seen": {"raw": True, "masked": True, "ai": False},
        "seq": {"raw": 100, "masked": 100, "ai": 0},
    }


class WatchdogClassifierTests(unittest.TestCase):
    def test_missing_status_is_treated_as_not_ready(self):
        # No heartbeat has ever arrived (e.g. right after worker
        # restart).  We must not force recovery yet.
        self.assertTrue(
            _watchdog_worker_legitimately_not_ready(
                status=None, status_age=float("inf"), mode=0, camera_index=0,
            )
        )

    def test_stale_status_is_treated_as_not_ready(self):
        status = _ready_status()
        # Heartbeat arrived 10 s ago -> stale, do not classify against
        # it.
        self.assertTrue(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=10.0, mode=0, camera_index=0,
            )
        )

    def test_worker_on_different_camera_is_not_ready(self):
        status = _ready_status(camera_index=0)
        # Our stream is for camera index 2 but worker is still on 0
        # (select_camera in flight).
        self.assertTrue(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=0.5, mode=0, camera_index=2,
            )
        )

    def test_device_opening_is_not_ready(self):
        status = _ready_status()
        status["device_state"] = "opening"
        self.assertTrue(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=0.5, mode=0, camera_index=0,
            )
        )

    def test_device_failed_is_not_ready(self):
        status = _ready_status()
        status["device_state"] = "failed"
        self.assertTrue(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=0.5, mode=0, camera_index=0,
            )
        )

    def test_recent_select_is_not_ready(self):
        status = _ready_status()
        # select_camera was applied half a second ago -- worker might
        # still be on its first capture iteration.
        status["select_applied_mono"] = time.monotonic() - 0.3
        self.assertTrue(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=0.5, mode=0, camera_index=0,
            )
        )

    def test_recent_capture_failure_is_not_ready(self):
        status = _ready_status()
        now = time.monotonic()
        # Capture failed 0.5 s ago; last success was longer ago.  USB
        # hiccup in progress, worker is retrying.
        status["last_capture_fail_mono"] = now - 0.5
        status["last_capture_ok_mono"] = now - 5.0
        self.assertTrue(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=0.5, mode=0, camera_index=0,
            )
        )

    def test_old_capture_failure_does_not_excuse_stall(self):
        status = _ready_status()
        now = time.monotonic()
        # Failure is ancient, capture has since recovered.
        status["last_capture_fail_mono"] = now - 10.0
        status["last_capture_ok_mono"] = now - 0.05
        self.assertFalse(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=0.5, mode=0, camera_index=0,
            )
        )

    def test_ai_inference_in_progress_is_not_ready_for_ai_mode(self):
        status = _ready_status()
        now = time.monotonic()
        # started > completed and started is >0.5 s ago -> YOLO is
        # blocked (warm-up or slow inference).
        status["ai_inference_started_mono"] = now - 1.5
        status["ai_inference_completed_mono"] = now - 5.0
        self.assertTrue(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=0.5, mode=3, camera_index=0,
            )
        )

    def test_ai_inference_in_progress_does_not_excuse_non_ai_stall(self):
        status = _ready_status()
        now = time.monotonic()
        # Same blocked AI inference, but browser is streaming Raw --
        # the AI pipeline doesn't matter.  If Raw is stuck that's a
        # real desync.
        status["ai_inference_started_mono"] = now - 1.5
        status["ai_inference_completed_mono"] = now - 5.0
        self.assertFalse(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=0.5, mode=0, camera_index=0,
            )
        )

    def test_ai_model_loading_is_not_ready_for_ai_mode(self):
        status = _ready_status()
        status["ai_model_state"] = "loading"
        self.assertTrue(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=0.5, mode=3, camera_index=0,
            )
        )

    def test_ai_model_not_loaded_is_not_ready_for_ai_mode(self):
        status = _ready_status()
        status["ai_model_state"] = "not_loaded"
        self.assertTrue(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=0.5, mode=3, camera_index=0,
            )
        )

    def test_fully_ready_but_stuck_triggers_recovery(self):
        # The watchdog's whole reason for existing: worker says
        # everything is fine, yet the relay has not seen a fresh frame
        # for >1 s.  Classifier must return False so recovery fires.
        status = _ready_status()
        self.assertFalse(
            _watchdog_worker_legitimately_not_ready(
                status=status, status_age=0.5, mode=1, camera_index=0,
            )
        )


if __name__ == "__main__":
    unittest.main()
