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
from fastapi import HTTPException


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
        def __init__(self, model_path=None, *args, **kwargs):
            self.model_path = str(model_path) if model_path is not None else None
            self.to_calls = []
            self.predict_calls = []
            self.call_calls = []

        def to(self, *, device: str):
            if self.model_path is not None and "ncnn_model" in self.model_path:
                raise TypeError("Exported NCNN models should not receive model.to().")
            self.to_calls.append(device)
            return self

        def __call__(self, image, **kwargs):
            self.call_calls.append((image, dict(kwargs)))
            return self.predict(source=image, **kwargs)

        def predict(self, source=None, **kwargs):
            self.predict_calls.append((source, dict(kwargs)))
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


def _make_fake_cameradevice_module():
    """Create a fake cameradevice module for testing."""
    cameradevice = types.ModuleType("server.cameradevice")

    class FakeCameraDevice:
        def __init__(self, **kwargs):
            self._settings = kwargs.get("settings", {})
        def open(self):
            return True
        def close(self):
            pass
        def get_image(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def get_properties(self):
            return {
                "flip_horizontal": False, "flip_vertical": False,
                "rotate": 0, "crop_top": 0.0, "crop_left": 0.0,
                "crop_bottom": 0.0, "crop_right": 0.0,
                "stretch_enabled": False, "stretch_width": 0,
                "stretch_height": 0, "mask_polygons": [],
                "width": 640, "height": 480, "fps": 30,
                "bitrate": 4000, "buffer_size": 1,
                "brightness": 128, "contrast": 32, "hue": 0,
                "saturation": 64, "sharpness": 0, "gamma": 100,
                "white_balance_temperature": 4500, "backlight": 0,
                "gain": 0, "focus": 0, "exposure": -6,
                "auto_white_balance_temperature": True,
                "auto_focus": True, "auto_exposure": True,
                "static_reticle_x": 0.5, "static_reticle_y": 0.5,
                "static_reticle_color": "#88ff00cc",
                "static_reticle_size": 1.0,
            }
        def set_properties(self, settings):
            self._settings = settings
        def get_supported_resolutions(self):
            return [{"width": 640, "height": 480, "label": "640 x 480"}]
        def get_capabilities(self):
            return {"exposure": True}

    cameradevice.CameraDevice = FakeCameraDevice
    cameradevice.CameraCV2Device = FakeCameraDevice
    cameradevice.CameraRPIDevice = FakeCameraDevice
    cameradevice.CameraDummyDevice = FakeCameraDevice
    cameradevice.create_camera_device = lambda **kwargs: FakeCameraDevice(**kwargs)
    return cameradevice


def _make_fake_transport_module():
    """Create a fake cameraframetransport module for testing."""
    transport_module = types.ModuleType("server.cameraframetransport")

    class FakeFrameTransport:
        def __init__(self, **kwargs):
            pass
        def get_worker_init_args(self):
            return {
                "shm_names": ("fake_r", "fake_m", "fake_a"),
                "shm_max_data_size": 1920 * 1080 * 3,
                "slot_locks": (mock.Mock(), mock.Mock(), mock.Mock()),
                "pose_pipe_conn": mock.Mock(),
                "command_pipe_conn": mock.Mock(),
            }
        def send_command(self, msg):
            pass
        def poll_pose(self, timeout=0):
            return False
        def recv_pose(self):
            return None
        def cleanup(self):
            pass
        @property
        def slot_raw(self):
            return FakeSlot()
        @property
        def slot_masked(self):
            return FakeSlot()
        @property
        def slot_ai(self):
            return FakeSlot()

    class FakeSlot:
        def read_header(self):
            return (0, 0, 0, 0.0, False, 0, 0)
        def read_frame(self):
            return (None, 0.0, False, 0)

    transport_module.FrameTransport = FakeFrameTransport
    transport_module.SharedFrameSlot = FakeSlot
    return transport_module


def _make_fake_worker_module():
    """Create a fake cameraworker module for testing."""
    worker_module = types.ModuleType("server.cameraworker")

    class FakeWorkerProcess:
        def __init__(self, **kwargs):
            self._alive = False
        def start(self):
            self._alive = True
        def is_alive(self):
            return self._alive
        def join(self, timeout=None):
            self._alive = False
        def terminate(self):
            self._alive = False
        def kill(self):
            self._alive = False

    worker_module.CameraWorkerProcess = FakeWorkerProcess
    return worker_module


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
        "server.cameradevice",
        "server.cameraframetransport",
        "server.cameraworker",
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
    cameraai.draw_pose = lambda image, pose, ai_setup, copy_image=True: image
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

    cameradevice = _make_fake_cameradevice_module()
    transport = _make_fake_transport_module()
    worker = _make_fake_worker_module()

    stub_modules = {
        "cv2": fake_cv2,
        "server.common": common,
        "server.ai_setup_constants": ai_setup_constants,
        "server.yolomodels": yolomodels,
        "server.cameraai": cameraai,
        "server.settingscontroller": settingscontroller,
        "server.aiagent": aiagent,
        "server.context": context,
        "server.cameradevice": cameradevice,
        "server.cameraframetransport": transport,
        "server.cameraworker": worker,
    }

    with mock.patch.dict(sys.modules, stub_modules):
        camera_module = importlib.import_module("server.camera")
        restapicameras = importlib.import_module("server.restapicameras")

    return camera_module, restapicameras


def _import_camerascontroller_module():
    _reset_modules(
        "server.camerascontroller",
        "server.camera",
        "server.settingscontroller",
        "server.cameradevice",
        "server.cameraframetransport",
        "server.cameraworker",
        "cv2",
    )

    fake_cv2 = types.ModuleType("cv2")

    camera_module = types.ModuleType("server.camera")

    class Camera:
        def __init__(self, **kwargs):
            self.camera_name = kwargs.get("camera_name", "test")
            self.kwargs = kwargs

        def stop(self):
            pass

    camera_module.Camera = Camera

    settings_payload = {"cameras": {}}
    settingscontroller = types.ModuleType("server.settingscontroller")
    settingscontroller.get_settings_sync = lambda: deepcopy(settings_payload)

    with mock.patch.dict(
        sys.modules,
        {
            "cv2": fake_cv2,
            "server.camera": camera_module,
            "server.settingscontroller": settingscontroller,
        },
    ):
        module = importlib.import_module("server.camerascontroller")

    module.CamerasController._singleton = None
    return module, settings_payload


def _make_camera(camera_module, **kwargs):
    """Create a Camera instance using the new constructor."""
    defaults = {
        "index": 0,
        "camera_index": 0,
        "camera_code": None,
        "camera_name": "test_camera",
        "camera_type": "dummy",
        "settings": {},
        "master_controller": SimpleNamespace(ai_agent=mock.Mock()),
    }
    defaults.update(kwargs)
    return camera_module.Camera(**defaults)


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
                preferred_device="cuda:0 (Nvidia GPU)",
            )
            second = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                preferred_device="cuda:0",
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
                preferred_device="arm_cpu",
            )
            cuda_model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                preferred_device="cuda:0",
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
                preferred_device="cuda:1 (Nvidia GPU)",
            )

        self.assertEqual(len(load_calls), 1)
        self.assertEqual(load_calls[0][1:3], ("pt", "cuda:1"))
        self.assertEqual(
            yolomodels.YOLOModels.normalize_device_value("cuda:1 (Nvidia GPU)"),
            "cuda:1 (Nvidia GPU)",
        )

    def test_normalize_device_value_keeps_supported_vulkan_labels(self):
        yolomodels = _import_yolomodels_module(cuda_available=False)

        self.assertEqual(
            yolomodels.YOLOModels.normalize_device_value("vulkan:0"),
            "vulkan:0 (AMD, Nvidia, ...)",
        )
        self.assertEqual(
            yolomodels.YOLOModels.normalize_device_value("vulkan:1 (AMD, Nvidia, ...)"),
            "vulkan:1 (AMD, Nvidia, ...)",
        )

    def test_get_model_uses_ncnn_cache_entries_for_distinct_vulkan_devices(self):
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
            vulkan0_model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                preferred_device="vulkan:0 (AMD, Nvidia, ...)",
            )
            vulkan1_model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                preferred_device="vulkan:1",
            )

        self.assertIsNot(vulkan0_model, vulkan1_model)
        self.assertEqual(len(load_calls), 2)
        self.assertEqual(load_calls[0][1:3], ("ncnn", "vulkan:0"))
        self.assertEqual(load_calls[1][1:3], ("ncnn", "vulkan:1"))

    def test_get_model_skips_model_to_for_arm_cpu_ncnn_models(self):
        yolomodels = _import_yolomodels_module(cuda_available=False)
        manager = yolomodels.YOLOModels()

        with mock.patch.object(yolomodels.Path, "exists", return_value=True):
            model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                preferred_device="arm_cpu",
            )

        self.assertIn("_ncnn_model", model.model_path)
        self.assertEqual(model.to_calls, [])

    def test_get_model_keeps_model_to_for_cuda_pt_models(self):
        yolomodels = _import_yolomodels_module(cuda_available=True, cuda_device_count=2, cuda_version="13.0")
        manager = yolomodels.YOLOModels()

        with mock.patch.object(yolomodels.Path, "exists", return_value=True):
            model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                preferred_device="cuda:1 (Nvidia GPU)",
            )

        self.assertTrue(model.model_path.endswith(".pt"))
        self.assertEqual(model.to_calls, ["cuda:1"])

    def test_predict_uses_direct_call_for_arm_cpu_ncnn_models(self):
        yolomodels = _import_yolomodels_module(cuda_available=False)
        manager = yolomodels.YOLOModels()
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with mock.patch.object(yolomodels.Path, "exists", return_value=True):
            results = manager.predict(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                preferred_device="arm_cpu",
                image=image,
                verbose=False,
            )
            model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                preferred_device="arm_cpu",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(model.call_calls), 1)
        self.assertEqual(model.call_calls[0][1], {"verbose": False})

    def test_predict_passes_vulkan_device_for_ncnn_models(self):
        yolomodels = _import_yolomodels_module(cuda_available=False)
        manager = yolomodels.YOLOModels()
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with mock.patch.object(yolomodels.Path, "exists", return_value=True):
            results = manager.predict(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                preferred_device="vulkan:0 (AMD, Nvidia, ...)",
                image=image,
                verbose=False,
            )
            model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                preferred_device="vulkan:0",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(model.to_calls, [])
        self.assertEqual(len(model.call_calls), 1)
        self.assertEqual(model.call_calls[0][1], {"verbose": False, "device": "vulkan:0"})

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
                preferred_device="cuda:0 (Nvidia GPU)",
            )
            cpu_model = manager.get_model(
                model_name=yolomodels.YOLOModels.DEFAULT_MODEL_NAME,
                preferred_device="cpu (optimized for x86)",
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
                        preferred_device="arm_cpu",
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
    def test_get_ai_setup_includes_vulkan_options_before_cuda(self):
        restapiaisetup = _import_aisetup_module(
            cuda_available=False,
            settings_payload={
                "aiSetup": {
                    "device": "vulkan:0",
                },
                "motors": [],
            },
        )

        response = asyncio.run(restapiaisetup.get_ai_setup())

        self.assertTrue(response["success"])
        self.assertEqual(response["aiSetup"]["device"], "vulkan:0 (AMD, Nvidia, ...)")
        self.assertEqual(
            [option["value"] for option in response["deviceOptions"]],
            [
                "cpu (optimized for x86)",
                "arm_cpu (optimized for ARM)",
                "vulkan:0 (AMD, Nvidia, ...)",
                "vulkan:1 (AMD, Nvidia, ...)",
                "cuda:0 (Nvidia GPU)",
                "cuda:1 (Nvidia GPU)",
            ],
        )

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
        camera = _make_camera(
            self.camera_module,
            camera_code="scope_camera",
            settings={"brightness": 0.2},
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
        camera = _make_camera(
            self.camera_module,
            camera_code="scope_camera",
            settings={"brightness": 0.2},
        )
        master_controller = self._make_master_controller([camera])

        response = asyncio.run(
            self.restapicameras.get_input_devices(master_controller=master_controller)
        )

        self.assertTrue(response["success"])
        self.assertIn("capabilities", response["input_devices"][0])
        self.assertNotIn("capabilities", camera.settings)

    def test_get_input_devices_restores_identity_fields_for_partial_settings(self):
        camera = _make_camera(
            self.camera_module,
            index=5,
            camera_index=1,
            camera_code="spotter_camera1",
            settings={"brightness": 0.2},
        )
        camera.settings = {
            "brightness": 0.2,
            "name": "   ",
            "supported_resolutions": [],
        }
        master_controller = self._make_master_controller([camera])

        response = asyncio.run(
            self.restapicameras.get_input_devices(master_controller=master_controller)
        )

        self.assertTrue(response["success"])
        device = response["input_devices"][0]
        self.assertEqual(device["index"], 5)
        self.assertEqual(device["camera_index"], 1)
        self.assertEqual(device["name"], camera.camera_name)
        self.assertEqual(device["supported_resolutions"], camera.supported_resolutions)
        self.assertIn("capabilities", device)

    def test_update_camera_reassigns_codes_and_applies_settings_copy(self):
        old_camera = _make_camera(
            self.camera_module,
            index=0,
            camera_index=0,
            camera_code="scope_camera",
            settings={"index": 0, "name": "Old camera"},
        )
        new_camera = _make_camera(
            self.camera_module,
            index=1,
            camera_index=1,
            camera_code=None,
            settings={"index": 1, "name": "New camera"},
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

    def test_update_camera_rejects_negative_device_index_with_404(self):
        camera = _make_camera(
            self.camera_module,
            camera_code="scope_camera",
            settings={"index": 0, "name": "Scope"},
        )
        master_controller = self._make_master_controller([camera])
        request = self.restapicameras.CameraUpdateRequest(index=-1, name="Invalid")

        with mock.patch.object(
            self.restapicameras.settingscontroller,
            "update_settings",
            new=mock.AsyncMock(),
        ) as update_settings_mock:
            with self.assertRaises(HTTPException) as exc_info:
                asyncio.run(
                    self.restapicameras.update_camera(
                        camera_code="scope_camera",
                        request=request,
                        master_controller=master_controller,
                    )
                )

        self.assertEqual(exc_info.exception.status_code, 404)
        update_settings_mock.assert_not_awaited()

    def test_reset_camera_rejects_invalid_configured_indexes_without_side_effects(self):
        camera = _make_camera(
            self.camera_module,
            camera_code="scope_camera",
            settings={"index": 0, "name": "Scope"},
        )
        camera.reset_settings = mock.Mock(wraps=camera.reset_settings)
        master_controller = self._make_master_controller([camera])
        invalid_settings_payloads = (
            {"cameras": {"scope_camera": {"index": -1}}},
            {"cameras": {}},
            {"cameras": {"scope_camera": {"index": 9}}},
        )

        for current_settings in invalid_settings_payloads:
            with self.subTest(current_settings=current_settings):
                camera.reset_settings.reset_mock()
                with (
                    mock.patch.object(
                        self.restapicameras.settingscontroller,
                        "get_settings",
                        new=mock.AsyncMock(return_value=deepcopy(current_settings)),
                    ),
                    mock.patch.object(
                        self.restapicameras.settingscontroller,
                        "update_settings",
                        new=mock.AsyncMock(),
                    ) as update_settings_mock,
                ):
                    with self.assertRaises(HTTPException) as exc_info:
                        asyncio.run(
                            self.restapicameras.reset_camera(
                                camera_code="scope_camera",
                                master_controller=master_controller,
                            )
                        )

                self.assertEqual(exc_info.exception.status_code, 404)
                camera.reset_settings.assert_not_called()
                update_settings_mock.assert_not_awaited()

    def test_stop_camera_rejects_disabled_camera_code_without_stopping_live_camera(self):
        scope_camera = _make_camera(
            self.camera_module,
            camera_code="scope_camera",
            settings={"index": 0, "name": "Scope"},
        )
        backup_camera = _make_camera(
            self.camera_module,
            index=1,
            camera_index=1,
            camera_code="spotter_camera1",
            settings={"index": 1, "name": "Backup"},
        )
        scope_camera.stop = mock.Mock()
        backup_camera.stop = mock.Mock()
        master_controller = self._make_master_controller([scope_camera, backup_camera])
        master_controller.cameras_controller.stop_camera = mock.Mock()

        with mock.patch.object(
            self.restapicameras.settingscontroller,
            "get_settings",
            new=mock.AsyncMock(return_value={"cameras": {"scope_camera": {"index": -1}}}),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                asyncio.run(
                    self.restapicameras.stop_camera(
                        camera_code="scope_camera",
                        master_controller=master_controller,
                    )
                )

        self.assertEqual(exc_info.exception.status_code, 404)
        master_controller.cameras_controller.stop_camera.assert_not_called()
        scope_camera.stop.assert_not_called()
        backup_camera.stop.assert_not_called()

    def test_stream_camera_rejects_unbound_camera_code_before_generating_frames(self):
        camera = _make_camera(
            self.camera_module,
            camera_code="scope_camera",
            settings={"index": 0, "name": "Scope"},
        )
        master_controller = self._make_master_controller([camera])

        with (
            mock.patch.object(
                self.restapicameras.settingscontroller,
                "get_settings",
                new=mock.AsyncMock(return_value={"cameras": {}}),
            ),
            mock.patch.object(
                self.restapicameras,
                "generate_camera_frames",
                new=mock.Mock(),
            ) as generate_camera_frames_mock,
        ):
            with self.assertRaises(HTTPException) as exc_info:
                asyncio.run(
                    self.restapicameras.stream_camera(
                        item_code="scope_camera",
                        master_controller=master_controller,
                    )
                )

        self.assertEqual(exc_info.exception.status_code, 404)
        generate_camera_frames_mock.assert_not_called()

    def test_stop_camera_offloads_blocking_stop_from_event_loop(self):
        camera = _make_camera(
            self.camera_module,
            camera_code="scope_camera",
            settings={"index": 0, "name": "Scope"},
        )
        stop_release = threading.Event()
        camera.stop = mock.Mock(side_effect=lambda: stop_release.wait(timeout=0.5))
        master_controller = self._make_master_controller([camera])

        with mock.patch.object(
            self.restapicameras.settingscontroller,
            "get_settings",
            new=mock.AsyncMock(return_value={"cameras": {"scope_camera": {"index": 0}}}),
        ):
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
            _make_camera(
                self.camera_module,
                index=i,
                camera_index=i,
                camera_code="scope_camera" if i == 0 else None,
                settings={"index": i, "name": f"Camera {i}"},
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


class CamerasControllerLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camerascontroller_module, self.settings_payload = _import_camerascontroller_module()

    def _make_controller(self, *, cameras):
        controller = object.__new__(self.camerascontroller_module.CamerasController)
        controller._master_controller = SimpleNamespace()
        controller._cameras = list(cameras)
        controller._lifecycle_lock = threading.RLock()
        return controller

    def test_cameras_controller_stop_stops_cached_cameras_without_rebuilding_list(self):
        class TrackedCamera:
            def __init__(self, camera_name: str):
                self.camera_name = camera_name
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1

        cameras = [TrackedCamera("scope"), TrackedCamera("spotter")]
        controller = self._make_controller(cameras=cameras)
        original_list = controller._cameras

        controller.stop()

        self.assertIs(controller._cameras, original_list)
        self.assertEqual([camera.stop_calls for camera in cameras], [1, 1])
        self.assertEqual(controller.cameras, cameras)

    def test_cameras_controller_reset_uses_shared_stop_phase_before_rebuilding(self):
        class TrackedCamera:
            def __init__(self, camera_name: str):
                self.camera_name = camera_name
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1

        old_cameras = [TrackedCamera("scope"), TrackedCamera("spotter")]
        controller = self._make_controller(cameras=old_cameras)

        created_cameras = []

        class CreatedCamera:
            def __init__(self, **kwargs):
                self.camera_name = kwargs.get("camera_name", "test")
                self.kwargs = kwargs
                self.stop_calls = 0
                created_cameras.append(self)

            def stop(self):
                self.stop_calls += 1

        self.camerascontroller_module.Camera = CreatedCamera
        stop_helper = mock.Mock(wraps=controller._stop_cameras_locked)
        controller._stop_cameras_locked = stop_helper

        with mock.patch.object(self.camerascontroller_module.time, "sleep", return_value=None):
            controller.reset(max_index=-5, reset_to_default=True)

        stop_helper.assert_called_once_with()
        self.assertEqual([camera.stop_calls for camera in old_cameras], [1, 1])
        self.assertEqual(len(created_cameras), 1)
        self.assertEqual(controller._cameras, created_cameras)
        self.assertTrue(all(camera not in old_cameras for camera in controller._cameras))


if __name__ == "__main__":
    unittest.main()
