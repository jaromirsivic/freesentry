import importlib
import sys
import types
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _reset_modules(*module_names: str) -> None:
    for module_name in module_names:
        sys.modules.pop(module_name, None)


def _import_restapiosmanagement_module():
    _reset_modules("server.restapiosmanagement", "server.common")

    common = types.ModuleType("server.common")
    common.get_platform_info = lambda: {"operating_system_code": "test"}

    with mock.patch.dict(sys.modules, {"server.common": common}):
        return importlib.import_module("server.restapiosmanagement")


def _make_test_client():
    module = _import_restapiosmanagement_module()
    app = FastAPI()
    app.include_router(module.router)
    return module, TestClient(app)


class RestApiOSManagementTests(unittest.TestCase):
    def test_set_system_datetime_returns_success_payload(self):
        module, client = _make_test_client()

        with (
            client,
            mock.patch.object(
                module.osmanagement,
                "set_system_datetime",
                return_value=(True, "Date and time updated successfully"),
            ) as set_datetime_mock,
        ):
            response = client.post(
                "/api/system/datetime",
                json={
                    "year": 2026,
                    "month": 4,
                    "day": 4,
                    "hour": 12,
                    "minute": 34,
                    "second": 56,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"success": True, "message": "Date and time updated successfully"},
        )
        set_datetime_mock.assert_called_once_with(
            year=2026,
            month=4,
            day=4,
            hour=12,
            minute=34,
            second=56,
        )

    def test_set_system_datetime_returns_500_when_backend_reports_failure(self):
        module, client = _make_test_client()

        with (
            client,
            mock.patch.object(
                module.osmanagement,
                "set_system_datetime",
                return_value=(False, "Setting system date/time is unsupported on this host"),
            ),
        ):
            response = client.post(
                "/api/system/datetime",
                json={
                    "year": 2026,
                    "month": 4,
                    "day": 4,
                    "hour": 12,
                    "minute": 34,
                    "second": 56,
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Setting system date/time is unsupported on this host"},
        )

    def test_set_system_timezone_returns_success_payload(self):
        module, client = _make_test_client()

        with (
            client,
            mock.patch.object(
                module.osmanagement,
                "set_system_timezone",
                return_value=(True, "Timezone updated to Europe/Prague"),
            ) as set_timezone_mock,
        ):
            response = client.post(
                "/api/system/timezone",
                json={"timezone": "Europe/Prague"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"success": True, "message": "Timezone updated to Europe/Prague"},
        )
        set_timezone_mock.assert_called_once_with(timezone_name="Europe/Prague")

    def test_set_system_timezone_returns_500_when_backend_reports_failure(self):
        module, client = _make_test_client()

        with (
            client,
            mock.patch.object(
                module.osmanagement,
                "set_system_timezone",
                return_value=(False, "Setting system timezone is unsupported on this host"),
            ),
        ):
            response = client.post(
                "/api/system/timezone",
                json={"timezone": "Europe/Prague"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Setting system timezone is unsupported on this host"},
        )

    def test_reboot_system_returns_success_payload(self):
        module, client = _make_test_client()

        with (
            client,
            mock.patch.object(
                module.osmanagement,
                "reboot_system",
                return_value=(True, "System reboot initiated"),
            ) as reboot_mock,
        ):
            response = client.post("/api/system/reboot")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"success": True, "message": "System reboot initiated"},
        )
        reboot_mock.assert_called_once_with()

    def test_reboot_system_returns_500_when_backend_raises(self):
        module, client = _make_test_client()

        with (
            client,
            mock.patch.object(
                module.osmanagement,
                "reboot_system",
                side_effect=RuntimeError("reboot service unavailable"),
            ),
        ):
            response = client.post("/api/system/reboot")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "reboot service unavailable"})

    def test_set_system_datetime_and_timezone_returns_success_payload(self):
        module, client = _make_test_client()

        with (
            client,
            mock.patch.object(
                module.osmanagement,
                "set_system_timezone",
                return_value=(True, "Timezone updated to Europe/Prague"),
            ) as set_timezone_mock,
            mock.patch.object(
                module.osmanagement,
                "set_system_datetime",
                return_value=(True, "Date and time updated successfully"),
            ) as set_datetime_mock,
        ):
            response = client.post(
                "/api/system/datetime-and-timezone",
                json={
                    "year": 2026,
                    "month": 4,
                    "day": 4,
                    "hour": 12,
                    "minute": 34,
                    "second": 56,
                    "timezone": "Europe/Prague",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "message": "Timezone set to Europe/Prague and date/time updated successfully",
            },
        )
        set_timezone_mock.assert_called_once_with(timezone_name="Europe/Prague")
        set_datetime_mock.assert_called_once_with(
            year=2026,
            month=4,
            day=4,
            hour=12,
            minute=34,
            second=56,
        )

    def test_set_system_datetime_and_timezone_stops_when_timezone_update_fails(self):
        module, client = _make_test_client()

        with (
            client,
            mock.patch.object(
                module.osmanagement,
                "set_system_timezone",
                return_value=(False, "Timezone update denied"),
            ) as set_timezone_mock,
            mock.patch.object(
                module.osmanagement,
                "set_system_datetime",
            ) as set_datetime_mock,
        ):
            response = client.post(
                "/api/system/datetime-and-timezone",
                json={
                    "year": 2026,
                    "month": 4,
                    "day": 4,
                    "hour": 12,
                    "minute": 34,
                    "second": 56,
                    "timezone": "Europe/Prague",
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Failed to set timezone: Timezone update denied"},
        )
        set_timezone_mock.assert_called_once_with(timezone_name="Europe/Prague")
        set_datetime_mock.assert_not_called()

    def test_set_system_datetime_and_timezone_returns_500_when_datetime_update_fails(self):
        module, client = _make_test_client()

        with (
            client,
            mock.patch.object(
                module.osmanagement,
                "set_system_timezone",
                return_value=(True, "Timezone updated to Europe/Prague"),
            ) as set_timezone_mock,
            mock.patch.object(
                module.osmanagement,
                "set_system_datetime",
                return_value=(False, "Date/time update denied"),
            ) as set_datetime_mock,
        ):
            response = client.post(
                "/api/system/datetime-and-timezone",
                json={
                    "year": 2026,
                    "month": 4,
                    "day": 4,
                    "hour": 12,
                    "minute": 34,
                    "second": 56,
                    "timezone": "Europe/Prague",
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Timezone updated but failed to set date/time: Date/time update denied"},
        )
        set_timezone_mock.assert_called_once_with(timezone_name="Europe/Prague")
        set_datetime_mock.assert_called_once_with(
            year=2026,
            month=4,
            day=4,
            hour=12,
            minute=34,
            second=56,
        )


if __name__ == "__main__":
    unittest.main()
