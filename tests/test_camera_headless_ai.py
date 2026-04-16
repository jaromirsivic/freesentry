"""Regression tests for scope_camera AI overlay publication.

These tests verify that:
1. ``_read_frames_from_transport`` never publishes a raw (un-overlaid) AI
   image to ``_frame_masked_ai`` for scope_camera — it must stage the image
   instead.
2. ``_process_pose`` always calls ``engage`` + ``draw_engagement_result``
   and publishes to ``_frame_masked_ai`` whenever a new AI image has been
   staged, even if the pose message from the worker is ``None`` (i.e. YOLO
   did not detect any keypoints).

The Camera class is instantiated via ``object.__new__`` to bypass its
hardware-probing ``__init__``; we wire only the attributes the two methods
actually need.
"""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from server import camera as camera_module
from server.common import Frame


def _make_camera(*, camera_code: str, ai_agent=None) -> camera_module.Camera:
    """Construct a Camera with only the attributes required by
    ``_read_frames_from_transport`` and ``_process_pose``. Bypasses
    ``__init__`` so we do not have to probe any camera hardware."""
    cam = object.__new__(camera_module.Camera)
    cam._camera_code = camera_code
    cam._camera_name = "test_camera"
    cam._transport = None
    cam._master_controller = SimpleNamespace(ai_agent=ai_agent) if ai_agent else None
    cam._frame_masked_ai = Frame(valid=False, image=np.zeros((4, 4, 3), dtype=np.uint8), time=0.0)
    cam._lock_frame_masked_ai = threading.Lock()
    cam._last_seq_ai = 0
    cam._staged_ai_image = None
    cam._staged_ai_time = 0.0
    cam._staged_ai_valid = False
    return cam


class _FakeSlot:
    def __init__(self, *, image, ts, valid=True, seq=1):
        self._image = image
        self._ts = ts
        self._valid = valid
        self._seq = seq

    def read_header(self):
        # Shape matches slot.read_header() in cameraframetransport.py:
        # (uid, rows, cols, channels, dtype, seq, data_size)
        return (0, 0, 0, 0, 0, self._seq, 0)

    def read_frame(self):
        return self._image, self._ts, self._valid, self._seq


class _FakeTransport:
    """Minimal transport that yields one new AI frame and an optional pose
    queue. ``slot_raw``/``slot_masked`` are fixed to the stale sequence so
    the raw/masked reads are short-circuited."""

    def __init__(self, *, ai_image, ai_ts, ai_seq, pose_queue=None):
        self.slot_raw = _FakeSlot(image=None, ts=0.0, valid=False, seq=0)
        self.slot_masked = _FakeSlot(image=None, ts=0.0, valid=False, seq=0)
        self.slot_ai = _FakeSlot(image=ai_image, ts=ai_ts, valid=True, seq=ai_seq)
        self._pose_queue = list(pose_queue or [])

    def poll_pose(self):
        return bool(self._pose_queue)

    def recv_pose(self):
        return self._pose_queue.pop(0)


class ReadFramesStagingTests(unittest.TestCase):
    """_read_frames_from_transport must stage AI frames for scope_camera
    and publish them directly for other cameras."""

    def test_scope_camera_stages_ai_frame_without_publishing(self):
        cam = _make_camera(camera_code="scope_camera")
        image = np.full((8, 8, 3), 123, dtype=np.uint8)
        cam._transport = _FakeTransport(ai_image=image, ai_ts=1.5, ai_seq=7)

        cam._read_frames_from_transport()

        self.assertTrue(cam._staged_ai_valid)
        self.assertIs(cam._staged_ai_image, image)
        self.assertEqual(cam._staged_ai_time, 1.5)
        self.assertEqual(cam._last_seq_ai, 7)
        # _frame_masked_ai must not have been overwritten with the raw image.
        self.assertFalse(cam._frame_masked_ai.valid)

    def test_non_scope_camera_publishes_ai_frame_directly(self):
        cam = _make_camera(camera_code="spotter_camera1")
        image = np.full((8, 8, 3), 42, dtype=np.uint8)
        cam._transport = _FakeTransport(ai_image=image, ai_ts=2.0, ai_seq=3)

        cam._read_frames_from_transport()

        self.assertFalse(cam._staged_ai_valid)
        self.assertTrue(cam._frame_masked_ai.valid)
        self.assertIs(cam._frame_masked_ai.image, image)
        self.assertEqual(cam._frame_masked_ai.time, 2.0)
        self.assertEqual(cam._last_seq_ai, 3)


class ProcessPoseOverlayTests(unittest.TestCase):
    """_process_pose must always run engage()+draw_engagement_result() and
    publish to _frame_masked_ai when a staged AI image is available,
    regardless of whether a pose was received."""

    def _make_ai_agent(self):
        return SimpleNamespace(
            engage=mock.Mock(return_value=SimpleNamespace(status="ok")),
            draw_engagement_result=mock.Mock(),
        )

    def test_overlay_drawn_when_pose_is_none(self):
        ai_agent = self._make_ai_agent()
        cam = _make_camera(camera_code="scope_camera", ai_agent=ai_agent)
        image = np.full((8, 8, 3), 7, dtype=np.uint8)
        cam._staged_ai_image = image
        cam._staged_ai_time = 9.0
        cam._staged_ai_valid = True
        cam._transport = _FakeTransport(
            ai_image=image, ai_ts=9.0, ai_seq=1,
            pose_queue=[{"type": "pose", "pose": None, "seq": 1}],
        )

        with mock.patch("server.camera.get_settings_sync", return_value={"aiSetup": {}}):
            cam._process_pose()

        ai_agent.engage.assert_called_once()
        ai_agent.draw_engagement_result.assert_called_once()
        # The frame passed to engage() must carry pose=None but the AI image.
        frame_arg = ai_agent.engage.call_args.kwargs["frame"]
        self.assertIsNone(frame_arg.pose)
        self.assertIs(frame_arg.image, image)
        # _frame_masked_ai must now hold the overlaid frame.
        self.assertTrue(cam._frame_masked_ai.valid)
        self.assertIs(cam._frame_masked_ai.image, image)
        # Staging slot must be consumed.
        self.assertFalse(cam._staged_ai_valid)
        self.assertIsNone(cam._staged_ai_image)

    def test_overlay_drawn_with_pose(self):
        ai_agent = self._make_ai_agent()
        cam = _make_camera(camera_code="scope_camera", ai_agent=ai_agent)
        image = np.full((8, 8, 3), 5, dtype=np.uint8)
        cam._staged_ai_image = image
        cam._staged_ai_time = 3.0
        cam._staged_ai_valid = True
        raw_pose = [{"keypoint": "stub"}]
        cam._transport = _FakeTransport(
            ai_image=image, ai_ts=3.0, ai_seq=4,
            pose_queue=[{"type": "pose", "pose": raw_pose, "seq": 4}],
        )

        translated_pose = [{"organ": "stub"}]
        with mock.patch("server.camera.get_settings_sync", return_value={"aiSetup": {}}):
            with mock.patch(
                "server.cameraai.translate_raw_pose_to_pose_dict",
                return_value=translated_pose,
            ) as translate_mock:
                cam._process_pose()

        translate_mock.assert_called_once()
        ai_agent.engage.assert_called_once()
        ai_agent.draw_engagement_result.assert_called_once()
        frame_arg = ai_agent.engage.call_args.kwargs["frame"]
        self.assertIs(frame_arg.pose, translated_pose)
        self.assertTrue(cam._frame_masked_ai.valid)

    def test_does_nothing_when_no_staged_frame(self):
        ai_agent = self._make_ai_agent()
        cam = _make_camera(camera_code="scope_camera", ai_agent=ai_agent)
        cam._transport = _FakeTransport(
            ai_image=np.zeros((1, 1, 3), dtype=np.uint8), ai_ts=0.0, ai_seq=0,
            pose_queue=[{"type": "pose", "pose": None, "seq": 0}],
        )
        # No staged image.
        cam._staged_ai_valid = False
        cam._staged_ai_image = None

        with mock.patch("server.camera.get_settings_sync", return_value={"aiSetup": {}}):
            cam._process_pose()

        ai_agent.engage.assert_not_called()
        ai_agent.draw_engagement_result.assert_not_called()
        self.assertFalse(cam._frame_masked_ai.valid)

    def test_non_scope_camera_only_drains_pose_pipe(self):
        ai_agent = self._make_ai_agent()
        cam = _make_camera(camera_code="spotter_camera1", ai_agent=ai_agent)
        cam._staged_ai_image = np.zeros((1, 1, 3), dtype=np.uint8)
        cam._staged_ai_valid = True
        cam._transport = _FakeTransport(
            ai_image=np.zeros((1, 1, 3), dtype=np.uint8), ai_ts=0.0, ai_seq=0,
            pose_queue=[{"type": "pose", "pose": None, "seq": 0}],
        )

        with mock.patch("server.camera.get_settings_sync", return_value={"aiSetup": {}}):
            cam._process_pose()

        ai_agent.engage.assert_not_called()
        ai_agent.draw_engagement_result.assert_not_called()
        # Staging remains untouched (non-scope path ignores it).
        self.assertTrue(cam._staged_ai_valid)


if __name__ == "__main__":
    unittest.main()
