import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from server import camera as camera_module
from server import common


class FakeCamera(camera_module.Camera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active = True
        self._camera = object()

    def _open(self) -> bool:
        self._active = True
        self._camera = object()
        return True

    def _close(self):
        self._active = False
        self._camera = None

    def _get_supported_resolutions(self):
        return [{"width": 640, "height": 480}]

    def _get_capabilities(self) -> dict:
        return {"exposure": True}

    def _get_camera_properties(self) -> dict:
        return {
            "index": self._index,
            "name": self._camera_name,
            "brightness": 0.0,
            "flip_horizontal": False,
            "flip_vertical": False,
            "rotate": 0,
            "crop_top": 0.0,
            "crop_left": 0.0,
            "crop_bottom": 0.0,
            "crop_right": 0.0,
            "stretch_enabled": False,
            "stretch_width": 0,
            "stretch_height": 0,
            "mask_polygons": [],
        }

    def _set_camera_properties(self, value: dict | None):
        return None

    def _get_image_ndarray(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)


class CameraHeadlessAITests(unittest.TestCase):
    def _make_ai_agent(self, *, activated: bool):
        return SimpleNamespace(
            is_fully_activated=lambda: activated,
            engage=mock.Mock(return_value=SimpleNamespace(status="engaging")),
            draw_engagement_result=mock.Mock(),
        )

    def test_scope_camera_keeps_capture_alive_while_ai_is_fully_activated(self):
        ai_agent = self._make_ai_agent(activated=True)
        camera = FakeCamera(
            index=0,
            camera_index=0,
            camera_code="scope_camera",
            settings={},
            master_controller=SimpleNamespace(ai_agent=ai_agent),
        )
        camera._last_access_time_raw_frame = time.time() - camera.TIMEOUT_SECONDS - 1.0

        self.assertTrue(camera._should_capture_next_frame())

    def test_scope_camera_still_times_out_when_ai_is_not_fully_activated(self):
        ai_agent = self._make_ai_agent(activated=False)
        camera = FakeCamera(
            index=0,
            camera_index=0,
            camera_code="scope_camera",
            settings={},
            master_controller=SimpleNamespace(ai_agent=ai_agent),
        )
        camera._last_access_time_raw_frame = time.time() - camera.TIMEOUT_SECONDS - 1.0

        self.assertFalse(camera._should_capture_next_frame())

    def test_scope_camera_continues_engagement_without_recent_ai_frame_consumers_when_active(self):
        ai_agent = self._make_ai_agent(activated=True)
        camera = FakeCamera(
            index=0,
            camera_index=0,
            camera_code="scope_camera",
            settings={},
            master_controller=SimpleNamespace(ai_agent=ai_agent),
        )
        stale_time = time.time() - camera.TIMEOUT_SECONDS - 1.0
        camera._last_access_time_masked_frame = stale_time
        camera._last_access_time_masked_ai_frame = stale_time

        with mock.patch(
            "server.camera.get_settings_sync",
            return_value={"aiSetup": {}, "cameras": {}},
        ):
            with mock.patch.object(
                camera,
                "_mask_ai_image",
                return_value=common.Frame(
                    valid=True,
                    image=np.zeros((4, 4, 3), dtype=np.uint8),
                    time=time.time(),
                    pose=None,
                ),
            ):
                camera._get_frame()

        ai_agent.engage.assert_called_once()
        ai_agent.draw_engagement_result.assert_called_once()

    def test_scope_camera_skips_engagement_without_recent_ai_frame_consumers_when_inactive(self):
        ai_agent = self._make_ai_agent(activated=False)
        camera = FakeCamera(
            index=0,
            camera_index=0,
            camera_code="scope_camera",
            settings={},
            master_controller=SimpleNamespace(ai_agent=ai_agent),
        )
        stale_time = time.time() - camera.TIMEOUT_SECONDS - 1.0
        camera._last_access_time_masked_frame = stale_time
        camera._last_access_time_masked_ai_frame = stale_time

        with mock.patch.object(camera, "_mask_ai_image") as mask_ai_mock:
            camera._get_frame()

        mask_ai_mock.assert_not_called()
        ai_agent.engage.assert_not_called()
        ai_agent.draw_engagement_result.assert_not_called()

    def test_explicit_stop_invalidates_stale_stream_token_before_camera_reopens(self):
        ai_agent = self._make_ai_agent(activated=False)
        camera = FakeCamera(
            index=0,
            camera_index=0,
            camera_code="scope_camera",
            settings={},
            master_controller=SimpleNamespace(ai_agent=ai_agent),
        )
        stream_token = camera.create_stream_token()
        camera._active = False
        camera._camera = None

        with mock.patch.object(camera, "is_alive", return_value=False):
            with mock.patch.object(camera, "start") as start_mock:
                camera.stop()
                frame = camera.get_stream_frame(mode=0, stream_token=stream_token)

        self.assertIsNone(frame)
        start_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
