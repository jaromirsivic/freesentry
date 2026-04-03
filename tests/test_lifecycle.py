import importlib
import sys
import threading
import time
import types
import unittest
from unittest import mock

from fastapi import APIRouter
from fastapi.testclient import TestClient
from starlette.responses import Response


class DummyStaticFiles:
    def __init__(self, *args, **kwargs):
        pass

    async def __call__(self, scope, receive, send):
        response = Response(status_code=404)
        await response(scope, receive, send)


class FakeMasterController:
    init_count = 0
    start_count = 0
    stop_count = 0
    last_instance = None

    def __init__(self):
        type(self).init_count += 1
        type(self).last_instance = self

    def start(self):
        type(self).start_count += 1

    def stop(self):
        type(self).stop_count += 1


def _make_router_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.router = APIRouter()
    return module


def _make_stub_modules() -> dict[str, types.ModuleType]:
    async def get_settings():
        return {}

    async def clear_cached_settings():
        return {"success": True}

    settingscontroller = types.ModuleType("server.settingscontroller")
    settingscontroller.get_settings = get_settings
    settingscontroller.clear_cached_settings = clear_cached_settings

    common = types.ModuleType("server.common")
    common.get_platform_info = lambda: {"operating_system_code": "test"}

    mastercontroller = types.ModuleType("server.mastercontroller")
    mastercontroller.MasterController = FakeMasterController

    modules = {
        "server.settingscontroller": settingscontroller,
        "server.common": common,
        "server.mastercontroller": mastercontroller,
    }

    for name in (
        "server.restapimotors",
        "server.restapicameras_old",
        "server.restapicameras",
        "server.restapisettings",
        "server.restapihotzone",
        "server.restapimanualcontrol",
        "server.restapiaisetup",
        "server.restapiosmanagement",
    ):
        modules[name] = _make_router_module(name)

    return modules


def _reset_fake_master_controller() -> None:
    FakeMasterController.init_count = 0
    FakeMasterController.start_count = 0
    FakeMasterController.stop_count = 0
    FakeMasterController.last_instance = None


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_fake_master_controller()
        for module_name in (
            "server.main",
            "server.settingscontroller",
            "server.common",
            "server.mastercontroller",
            "server.restapimotors",
            "server.restapicameras_old",
            "server.restapicameras",
            "server.restapisettings",
            "server.restapihotzone",
            "server.restapimanualcontrol",
            "server.restapiaisetup",
            "server.restapiosmanagement",
            "server.restapihealth",
        ):
            sys.modules.pop(module_name, None)

    def _import_main(self):
        with mock.patch("fastapi.staticfiles.StaticFiles", DummyStaticFiles):
            with mock.patch.dict(sys.modules, _make_stub_modules()):
                return importlib.import_module("server.main")

    def test_importing_main_does_not_construct_master_controller(self):
        module = self._import_main()

        self.assertEqual(FakeMasterController.init_count, 0)
        self.assertFalse(hasattr(module.app.state, "master_controller"))

    def test_lifespan_constructs_and_stops_master_controller_once(self):
        module = self._import_main()

        with TestClient(module.app):
            self.assertEqual(FakeMasterController.init_count, 1)
            self.assertEqual(FakeMasterController.start_count, 1)
            self.assertEqual(module.app.state.master_controller, FakeMasterController.last_instance)

        self.assertEqual(FakeMasterController.stop_count, 1)
        self.assertFalse(hasattr(module.app.state, "master_controller"))

    def test_lifespan_serves_api_while_deferred_startup_is_running(self):
        module = self._import_main()
        startup_entered = threading.Event()
        release_startup = threading.Event()

        def slow_wifi_startup(*, settings, os_code):
            startup_entered.set()
            release_startup.wait(timeout=1.0)

        with (
            mock.patch.object(module, "log_ai_runtime_diagnostics", return_value=None),
            mock.patch.object(module, "_wifi_startup_sync", side_effect=slow_wifi_startup),
            mock.patch.object(module, "_execute_startup_script_sync", return_value=None),
        ):
            started_at = time.monotonic()
            with TestClient(module.app) as client:
                self.assertLess(time.monotonic() - started_at, 0.3)
                self.assertTrue(startup_entered.wait(timeout=1.0))

                response = client.get("/api/health/readiness")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["phase"], "post_start")
                self.assertFalse(response.json()["ready"])

                release_startup.set()
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    response = client.get("/api/health/readiness")
                    if response.status_code == 200:
                        break
                    time.sleep(0.01)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["phase"], "ready")
                self.assertTrue(response.json()["ready"])

    def test_readiness_reports_failed_deferred_startup(self):
        module = self._import_main()

        with (
            mock.patch.object(module, "log_ai_runtime_diagnostics", return_value=None),
            mock.patch.object(module, "_wifi_startup_sync", return_value=None),
            mock.patch.object(
                module,
                "_execute_startup_script_sync",
                side_effect=RuntimeError("startup script failed"),
            ),
        ):
            with TestClient(module.app) as client:
                deadline = time.monotonic() + 1.0
                response = None
                while time.monotonic() < deadline:
                    response = client.get("/api/health/readiness")
                    if response.json()["phase"] == "failed":
                        break
                    time.sleep(0.01)

                self.assertIsNotNone(response)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["phase"], "failed")
                self.assertIn("startup script failed", response.json()["lastError"])

    def test_ai_agent_init_does_not_start_thread(self):
        aiagent = importlib.import_module("server.aiagent")

        with mock.patch.object(aiagent.AIAgent, "start", autospec=True) as start_mock:
            agent = aiagent.AIAgent(master_controller=object())

        self.assertFalse(start_mock.called)
        self.assertFalse(agent.is_alive())


if __name__ == "__main__":
    unittest.main()
