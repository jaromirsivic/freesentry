import asyncio
import importlib
import sys
import threading
import time
import types
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import numpy as np


def _reset_modules(*module_names: str) -> None:
    for module_name in module_names:
        sys.modules.pop(module_name, None)


def _make_fake_torch_module(
    *,
    cuda_available: bool,
    cuda_device_count: int | None = None,
    cuda_version: str | None = None,
):
    effective_device_count = (
        int(cuda_device_count)
        if cuda_device_count is not None
        else (1 if cuda_available else 0)
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.__file__ = "C:/fake/site-packages/torch/__init__.py"
    fake_torch.__version__ = "2.9.1"
    fake_torch.version = SimpleNamespace(cuda=cuda_version if cuda_available else None)
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: effective_device_count,
    )
    return fake_torch


def _make_fake_ultralytics_module():
    fake_ultralytics = types.ModuleType("ultralytics")

    class FakeYOLO:
        def __init__(self, *args, **kwargs):
            self.to_calls = []

        def to(self, *, device: str):
            self.to_calls.append(device)
            return self

        def __call__(self, image, **kwargs):
            return [SimpleNamespace(keypoints=None)]

    fake_ultralytics.YOLO = FakeYOLO
    return fake_ultralytics


def _import_yolomodels_module(
    *,
    cuda_available: bool = False,
    cuda_device_count: int | None = None,
    cuda_version: str | None = None,
):
    _reset_modules("server.yolomodels", "server.ai_setup_constants", "ultralytics", "torch")

    fake_ultralytics = _make_fake_ultralytics_module()
    fake_torch = _make_fake_torch_module(
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_version=cuda_version,
    )

    with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics, "torch": fake_torch}):
        return importlib.import_module("server.yolomodels")


def _import_aisetup_module(
    *,
    cuda_available: bool,
    cuda_device_count: int | None = None,
    cuda_version: str | None = None,
    settings_payload: dict | None = None,
):
    _reset_modules(
        "server.restapiaisetup",
        "server.settingscontroller",
        "server.yolomodels",
        "server.ai_setup_constants",
        "ultralytics",
        "torch",
    )

    fake_ultralytics = _make_fake_ultralytics_module()
    fake_torch = _make_fake_torch_module(
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_version=cuda_version,
    )

    payload = deepcopy(settings_payload) if settings_payload is not None else {"aiSetup": {}, "motors": []}
    settingscontroller = types.ModuleType("server.settingscontroller")

    async def get_settings():
        return deepcopy(payload)

    async def update_settings(mutator):
        updated_settings = deepcopy(payload)
        mutator(updated_settings)
        return None

    settingscontroller.get_settings = get_settings
    settingscontroller.update_settings = update_settings

    with mock.patch.dict(
        sys.modules,
        {
            "server.settingscontroller": settingscontroller,
            "ultralytics": fake_ultralytics,
            "torch": fake_torch,
        },
    ):
        return importlib.import_module("server.restapiaisetup")


def _import_camera_stack():
    _reset_modules(
        "server.restapicameras",
        "server.camera",
        "server.common",
        "server.ai_setup_constants",
        "server.yolomodels",
        "server.cameraai",
        "server.settingscontroller",
        "server.aiagent",
        "server.context",
        "cv2",
    )

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.FONT_HERSHEY_SIMPLEX = 0
    fake_cv2.ROTATE_90_COUNTERCLOCKWISE = 0
    fake_cv2.ROTATE_180 = 1
    fake_cv2.ROTATE_90_CLOCKWISE = 2
    fake_cv2.VideoCapture = object
    fake_cv2.typing = SimpleNamespace(MatLike=object)
    fake_cv2.getTextSize = lambda text, font, scale, thickness: ((10, 10), 0)
    fake_cv2.putText = lambda image, *args, **kwargs: image
    fake_cv2.flip = lambda image, mode: image
    fake_cv2.rotate = lambda image, mode: image
    fake_cv2.resize = lambda image, size: image
    fake_cv2.fillPoly = lambda image, pts, color: None
    fake_cv2.bitwise_and = lambda image1, image2, mask=None: image1

    common = types.ModuleType("server.common")

    class Frame:
        def __init__(self, *, valid: bool, image: np.ndarray, time: float, pose=None):
            self.valid = valid
            self.image = image
            self.time = time
            self.pose = pose

        def copy(self):
            return Frame(
                valid=self.valid,
                image=self.image.copy(),
                time=self.time,
                pose=deepcopy(self.pose),
            )

    common.Frame = Frame
    common.EPSILON_DELAY = 0.001

    ai_setup_constants = types.ModuleType("server.ai_setup_constants")
    ai_setup_constants.DEFAULT_DEVICE = "cpu"

    yolomodels = types.ModuleType("server.yolomodels")

    class StubYOLOModels:
        DEFAULT_MODEL_NAME = "Simple and Fast"

        @property
        def default_model_name(self) -> str:
            return self.DEFAULT_MODEL_NAME

        def get_model(self, *, model_name: str, device: str = "cpu"):
            return object()

        def predict(self, *, model_name: str, device: str = "cpu", image=None, **kwargs):
            return [SimpleNamespace(keypoints=None)]

    yolomodels.YOLOModels = StubYOLOModels

    cameraai = types.ModuleType("server.cameraai")
    cameraai.draw_pose = lambda image, pose, ai_setup: image
    cameraai.get_pose_dict = lambda keypoints, ai_setup: []
    cameraai.translate_raw_pose_to_pose_dict = lambda raw_pose, ai_setup: raw_pose

    settingscontroller = types.ModuleType("server.settingscontroller")

    def get_settings_sync():
        return {"aiSetup": {}}

    async def get_settings():
        return {}

    async def update_settings(mutator):
        return mutator({})

    async def clear_cached_settings():
        return {"success": True}

    settingscontroller.get_settings_sync = get_settings_sync
    settingscontroller.get_settings = get_settings
    settingscontroller.update_settings = update_settings
    settingscontroller.clear_cached_settings = clear_cached_settings

    aiagent = types.ModuleType("server.aiagent")

    class AIAgent:
        def __init__(self, *args, **kwargs):
            pass

        def engage(self, *, frame, settings):
            return None

        def draw_engagement_result(self, *, frame, engagement_result):
            return None

    aiagent.AIAgent = AIAgent

    context = types.ModuleType("server.context")
    context.get_master_controller = lambda: None

    stub_modules = {
        "cv2": fake_cv2,
        "server.common": common,
        "server.ai_setup_constants": ai_setup_constants,
        "server.yolomodels": yolomodels,
        "server.cameraai": cameraai,
        "server.settingscontroller": settingscontroller,
        "server.aiagent": aiagent,
        "server.context": context,
    }

    with mock.patch.dict(sys.modules, stub_modules):
        camera_module = importlib.import_module("server.camera")
        restapicameras = importlib.import_module("server.restapicameras")

    return camera_module, restapicameras


def _make_fake_camera_class(camera_module):
    class FakeCamera(camera_module.Camera):
        def __init__(self, *args, **kwargs):
            self.applied_settings = []
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
            self.applied_settings.append(deepcopy(value))

        def _get_image_ndarray(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    return FakeCamera


class YOLOModelsTests(unittest.TestCase):
    def test_get_model_reuses_cache_for_same_normalized_device(self):
        yolomodels = _import_yolomodels_module(cuda_available=True, cuda_device_count=2, cuda_version="13.0")
        manager = yolomodels.YOLOModels()
        load_calls = []

        def fake_load(_self, *, model_name, model_type, device="cpu", model_filename=None):
            load_calls.append((model_name, model_type, device, model_filename))
            return yolomodels._ModelCacheEntry(model=object(), inference_lock=threading.Lock())

        with mock.patch.object(
            yolomodels.YOLOModels,
            "_load_model_entry",
            autospec=True,
            side_effect=fake_load,
        ):
            first = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                device="cuda:0 (Nvidia GPU)",
            )
            second = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                device="cuda:0",
            )

        self.assertIs(first, second)
        self.assertEqual(len(load_calls), 1)
        self.assertEqual(load_calls[0][1], "pt")
        self.assertEqual(load_calls[0][2], "cuda:0")

    def test_get_model_separates_cache_entries_by_device(self):
        yolomodels = _import_yolomodels_module(cuda_available=True, cuda_device_count=2, cuda_version="13.0")
        manager = yolomodels.YOLOModels()
        load_calls = []

        def fake_load(_self, *, model_name, model_type, device="cpu", model_filename=None):
            load_calls.append((model_name, model_type, device, model_filename))
            return yolomodels._ModelCacheEntry(model=object(), inference_lock=threading.Lock())

        with mock.patch.object(
            yolomodels.YOLOModels,
            "_load_model_entry",
            autospec=True,
            side_effect=fake_load,
        ):
            arm_cpu_model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                device="arm_cpu",
            )
            cuda_model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                device="cuda:0",
            )

        self.assertIsNot(arm_cpu_model, cuda_model)
        self.assertEqual(len(load_calls), 2)
        self.assertEqual(load_calls[0][1:3], ("ncnn", "cpu"))
        self.assertEqual(load_calls[1][1:3], ("pt", "cuda:0"))

    def test_get_model_uses_second_cuda_device_when_available(self):
        yolomodels = _import_yolomodels_module(cuda_available=True, cuda_device_count=2, cuda_version="13.0")
        manager = yolomodels.YOLOModels()
        load_calls = []

        def fake_load(_self, *, model_name, model_type, device="cpu", model_filename=None):
            load_calls.append((model_name, model_type, device, model_filename))
            return yolomodels._ModelCacheEntry(model=object(), inference_lock=threading.Lock())

        with mock.patch.object(
            yolomodels.YOLOModels,
            "_load_model_entry",
            autospec=True,
            side_effect=fake_load,
        ):
            manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                device="cuda:1 (Nvidia GPU)",
            )

        self.assertEqual(len(load_calls), 1)
        self.assertEqual(load_calls[0][1:3], ("pt", "cuda:1"))
        self.assertEqual(
            yolomodels.YOLOModels.normalize_device_value("cuda:1 (Nvidia GPU)"),
            "cuda:1 (Nvidia GPU)",
        )

    def test_cuda_request_falls_back_to_cpu_and_reuses_cpu_cache_on_cpu_only_runtime(self):
        yolomodels = _import_yolomodels_module(cuda_available=False)
        manager = yolomodels.YOLOModels()
        load_calls = []

        def fake_load(_self, *, model_name, model_type, device="cpu", model_filename=None):
            load_calls.append((model_name, model_type, device, model_filename))
            return yolomodels._ModelCacheEntry(model=object(), inference_lock=threading.Lock())

        with mock.patch.object(
            yolomodels.YOLOModels,
            "_load_model_entry",
            autospec=True,
            side_effect=fake_load,
        ):
            cuda_model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                device="cuda:0 (Nvidia GPU)",
            )
            cpu_model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                device="cpu (optimized for x86)",
            )

        self.assertIs(cuda_model, cpu_model)
        self.assertEqual(len(load_calls), 1)
        self.assertEqual(load_calls[0][1:3], ("pt", "cpu"))
        self.assertEqual(
            yolomodels.YOLOModels.normalize_device_value("cuda:0 (Nvidia GPU)"),
            "cpu (optimized for x86)",
        )

    def test_predict_serializes_shared_model_inference(self):
        yolomodels = _import_yolomodels_module(cuda_available=False)
        manager = yolomodels.YOLOModels()

        class BlockingModel:
            def __init__(self):
                self.active_calls = 0
                self.max_active_calls = 0
                self._lock = threading.Lock()

            def __call__(self, image, **kwargs):
                with self._lock:
                    self.active_calls += 1
                    self.max_active_calls = max(self.max_active_calls, self.active_calls)
                try:
                    time.sleep(0.05)
                    return [SimpleNamespace(keypoints=None)]
                finally:
                    with self._lock:
                        self.active_calls -= 1

        blocking_model = BlockingModel()

        def fake_load(_self, *, model_name, model_type, device="cpu", model_filename=None):
            return yolomodels._ModelCacheEntry(
                model=blocking_model,
                inference_lock=threading.Lock(),
            )

        with mock.patch.object(
            yolomodels.YOLOModels,
            "_load_model_entry",
            autospec=True,
            side_effect=fake_load,
        ):
            start_barrier = threading.Barrier(2)
            errors = []

            def worker():
                try:
                    start_barrier.wait(timeout=2)
                    manager.predict(
                        model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                        device="arm_cpu",
                        image=np.zeros((2, 2, 3), dtype=np.uint8),
                        verbose=False,
                    )
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)

        self.assertFalse(errors, f"Unexpected inference errors: {errors}")
        self.assertEqual(blocking_model.max_active_calls, 1)


class AISetupDeviceTests(unittest.TestCase):
    def test_get_ai_setup_keeps_cuda_options_disabled_when_runtime_has_no_cuda(self):
        restapiaisetup = _import_aisetup_module(
            cuda_available=False,
            settings_payload={
                "aiSetup": {
                    "device": "cuda:0 (Nvidia GPU)",
                },
                "motors": [],
            },
        )

        response = asyncio.run(restapiaisetup.get_ai_setup())

        self.assertTrue(response["success"])
        self.assertEqual(response["aiSetup"]["device"], "cpu (optimized for x86)")
        self.assertEqual(response["defaultDevice"], "cpu (optimized for x86)")
        device_options = {option["value"]: option for option in response["deviceOptions"]}
        self.assertIn("cuda:0 (Nvidia GPU)", device_options)
        self.assertIn("cuda:1 (Nvidia GPU)", device_options)
        self.assertTrue(device_options["cuda:0 (Nvidia GPU)"]["disabled"])
        self.assertTrue(device_options["cuda:1 (Nvidia GPU)"]["disabled"])

    def test_get_ai_setup_disables_only_unavailable_cuda_devices(self):
        restapiaisetup = _import_aisetup_module(
            cuda_available=True,
            cuda_device_count=1,
            cuda_version="13.0",
            settings_payload={
                "aiSetup": {
                    "device": "cuda:1 (Nvidia GPU)",
                },
                "motors": [],
            },
        )

        response = asyncio.run(restapiaisetup.get_ai_setup())

        self.assertTrue(response["success"])
        self.assertEqual(response["aiSetup"]["device"], "cpu (optimized for x86)")
        device_options = {option["value"]: option for option in response["deviceOptions"]}
        self.assertFalse(device_options["cuda:0 (Nvidia GPU)"]["disabled"])
        self.assertTrue(device_options["cuda:1 (Nvidia GPU)"]["disabled"])


class CameraConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_module, self.restapicameras = _import_camera_stack()
        self.FakeCamera = _make_fake_camera_class(self.camera_module)

    def _make_master_controller(self, cameras):
        class FakeCamerasController:
            def __init__(self, initial_cameras):
                self._cameras = list(initial_cameras)

            @property
            def cameras(self):
                return list(self._cameras)

            def reset(self, *, reset_to_default=False):
                return None

            def stop_camera(self, *, index: int):
                self._cameras[index].stop()

        master_controller = SimpleNamespace(ai_agent=mock.Mock())
        for camera in cameras:
            camera._master_controller = master_controller
        master_controller.cameras_controller = FakeCamerasController(cameras)
        return master_controller

    def test_camera_settings_are_copy_on_write_and_keep_newer_updates_pending(self):
        camera = self.FakeCamera(
            index=0,
            camera_index=0,
            camera_code="scope_camera",
            settings={"brightness": 0.2},
            master_controller=SimpleNamespace(ai_agent=mock.Mock()),
        )

        external_settings = camera.settings
        external_settings["brightness"] = 0.9
        self.assertEqual(camera.settings["brightness"], 0.2)

        first_pending = camera._get_pending_settings_update()
        self.assertIsNotNone(first_pending)
        _, first_version = first_pending

        camera.settings = {**camera.settings, "brightness": 0.7}
        camera._mark_settings_applied(settings_version=first_version)

        second_pending = camera._get_pending_settings_update()
        self.assertIsNotNone(second_pending)
        second_settings, second_version = second_pending
        self.assertGreater(second_version, first_version)
        self.assertEqual(second_settings["brightness"], 0.7)

    def test_get_input_devices_does_not_leak_internal_settings(self):
        camera = self.FakeCamera(
            index=0,
            camera_index=0,
            camera_code="scope_camera",
            settings={"brightness": 0.2},
            master_controller=SimpleNamespace(ai_agent=mock.Mock()),
        )
        master_controller = self._make_master_controller([camera])

        response = asyncio.run(
            self.restapicameras.get_input_devices(master_controller=master_controller)
        )

        self.assertTrue(response["success"])
        self.assertIn("capabilities", response["input_devices"][0])
        self.assertNotIn("capabilities", camera.settings)

    def test_update_camera_reassigns_codes_and_applies_settings_copy(self):
        old_camera = self.FakeCamera(
            index=0,
            camera_index=0,
            camera_code="scope_camera",
            settings={"index": 0, "name": "Old camera"},
            master_controller=SimpleNamespace(ai_agent=mock.Mock()),
        )
        new_camera = self.FakeCamera(
            index=1,
            camera_index=1,
            camera_code=None,
            settings={"index": 1, "name": "New camera"},
            master_controller=SimpleNamespace(ai_agent=mock.Mock()),
        )
        master_controller = self._make_master_controller([old_camera, new_camera])
        persisted_settings = {}

        async def fake_update_settings(mutator):
            current_settings = {
                "cameras": {
                    "scope_camera": {"index": 0, "name": "Old camera"},
                    "spotter_camera1": {"index": 1, "name": "Spotter 1"},
                }
            }
            old_index = mutator(current_settings)
            persisted_settings.update(deepcopy(current_settings))
            return old_index

        request = self.restapicameras.CameraUpdateRequest(
            index=1,
            name="Scope moved",
            brightness=0.75,
        )

        with mock.patch.object(
            self.restapicameras.settingscontroller,
            "update_settings",
            new=mock.AsyncMock(side_effect=fake_update_settings),
        ):
            response = asyncio.run(
                self.restapicameras.update_camera(
                    camera_code="scope_camera",
                    request=request,
                    master_controller=master_controller,
                )
            )

        self.assertEqual(response, {"success": True})
        self.assertEqual(persisted_settings["cameras"]["scope_camera"]["index"], 1)
        self.assertEqual(persisted_settings["cameras"]["spotter_camera1"]["index"], -1)
        self.assertIsNone(old_camera.camera_code)
        self.assertEqual(new_camera.camera_code, "scope_camera")
        self.assertEqual(new_camera.settings["name"], "Scope moved")
        self.assertEqual(new_camera.settings["brightness"], 0.75)
        self.assertEqual(
            new_camera.settings["supported_resolutions"],
            new_camera.supported_resolutions,
        )

    def test_stop_camera_offloads_blocking_stop_from_event_loop(self):
        camera = self.FakeCamera(
            index=0,
            camera_index=0,
            camera_code="scope_camera",
            settings={"index": 0, "name": "Scope"},
            master_controller=SimpleNamespace(ai_agent=mock.Mock()),
        )
        stop_release = threading.Event()
        camera.stop = mock.Mock(side_effect=lambda: stop_release.wait(timeout=0.5))
        master_controller = self._make_master_controller([camera])

        async def scenario():
            stop_task = asyncio.create_task(
                self.restapicameras.stop_camera(
                    camera_code="scope_camera",
                    master_controller=master_controller,
                )
            )
            probe_task = asyncio.create_task(asyncio.sleep(0.01, result="probe"))
            started_at = time.monotonic()
            done, _ = await asyncio.wait(
                {stop_task, probe_task},
                timeout=0.05,
                return_when=asyncio.FIRST_COMPLETED,
            )
            elapsed = time.monotonic() - started_at

            self.assertLess(elapsed, 0.2)
            self.assertIn(probe_task, done)
            self.assertNotIn(stop_task, done)

            stop_release.set()
            result = await asyncio.wait_for(stop_task, timeout=1.0)
            self.assertEqual(result, {"success": True})

        asyncio.run(scenario())
        camera.stop.assert_called_once()

    def test_reset_all_cameras_offloads_blocking_reset_from_event_loop(self):
        cameras = [
            self.FakeCamera(
                index=i,
                camera_index=i,
                camera_code="scope_camera" if i == 0 else None,
                settings={"index": i, "name": f"Camera {i}"},
                master_controller=SimpleNamespace(ai_agent=mock.Mock()),
            )
            for i in range(4)
        ]
        master_controller = self._make_master_controller(cameras)
        reset_release = threading.Event()
        master_controller.cameras_controller.reset = mock.Mock(
            side_effect=lambda *, reset_to_default=False: reset_release.wait(timeout=0.5)
        )

        with (
            mock.patch.object(
                self.restapicameras.settingscontroller,
                "clear_cached_settings",
                new=mock.AsyncMock(return_value={"success": True}),
            ) as clear_cached_settings_mock,
            mock.patch.object(
                self.restapicameras,
                "reset_camera",
                new=mock.AsyncMock(return_value={"success": True}),
            ) as reset_camera_mock,
        ):
            async def scenario():
                reset_task = asyncio.create_task(
                    self.restapicameras.reset_all_cameras(master_controller=master_controller)
                )
                probe_task = asyncio.create_task(asyncio.sleep(0.01, result="probe"))
                started_at = time.monotonic()
                done, _ = await asyncio.wait(
                    {reset_task, probe_task},
                    timeout=0.05,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                elapsed = time.monotonic() - started_at

                self.assertLess(elapsed, 0.2)
                self.assertIn(probe_task, done)
                self.assertNotIn(reset_task, done)

                reset_release.set()
                result = await asyncio.wait_for(reset_task, timeout=1.0)
                self.assertEqual(result, {"success": True})

            asyncio.run(scenario())

        clear_cached_settings_mock.assert_awaited_once()
        self.assertEqual(reset_camera_mock.await_count, len(self.restapicameras.CAMERA_NAMES))
        master_controller.cameras_controller.reset.assert_called_once_with(reset_to_default=True)


if __name__ == "__main__":
    unittest.main()
