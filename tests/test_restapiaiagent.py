import asyncio
import importlib
import sys
import types
import unittest
from unittest import mock


def _reset_modules(*module_names: str) -> None:
    for module_name in module_names:
        sys.modules.pop(module_name, None)


def _import_aiagent_module():
    _reset_modules(
        "server.restapiaiagent",
        "server.context",
    )

    context = types.ModuleType("server.context")
    context.get_master_controller = lambda: None

    with mock.patch.dict(
        sys.modules,
        {
            "server.context": context,
        },
    ):
        module = importlib.import_module("server.restapiaiagent")

    return module


class FakeAIAgent:
    def __init__(self, *, activated: bool = False):
        self._activated = activated

    def is_fully_activated(self) -> bool:
        return self._activated

    def activate(self) -> None:
        self._activated = True

    def deactivate(self) -> None:
        self._activated = False


class AIAgentRuntimeApiTests(unittest.TestCase):
    def test_get_aiagent_activation_returns_current_runtime_state(self):
        module = _import_aiagent_module()
        master_controller = types.SimpleNamespace(ai_agent=FakeAIAgent(activated=True))

        response = asyncio.run(
            module.get_aiagent_activation(master_controller=master_controller)
        )

        self.assertEqual(
            response,
            {"success": True, "aiagent_fully_activated": True},
        )

    def test_activate_aiagent_enables_runtime_state(self):
        module = _import_aiagent_module()
        master_controller = types.SimpleNamespace(ai_agent=FakeAIAgent(activated=False))

        response = asyncio.run(
            module.activate_aiagent(master_controller=master_controller)
        )

        self.assertEqual(
            response,
            {"success": True, "aiagent_fully_activated": True},
        )

    def test_deactivate_aiagent_disables_runtime_state(self):
        module = _import_aiagent_module()
        master_controller = types.SimpleNamespace(ai_agent=FakeAIAgent(activated=True))

        response = asyncio.run(
            module.deactivate_aiagent(master_controller=master_controller)
        )

        self.assertEqual(
            response,
            {"success": True, "aiagent_fully_activated": False},
        )


if __name__ == "__main__":
    unittest.main()
