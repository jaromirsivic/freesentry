import importlib
import sys
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

    def test_ai_agent_init_does_not_start_thread(self):
        aiagent = importlib.import_module("server.aiagent")

        with mock.patch.object(aiagent.AIAgent, "start", autospec=True) as start_mock:
            agent = aiagent.AIAgent(master_controller=object())

        self.assertFalse(start_mock.called)
        self.assertFalse(agent.is_alive())


if __name__ == "__main__":
    unittest.main()
