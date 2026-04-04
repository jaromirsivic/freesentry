import unittest
from unittest import mock

from server import osmanagement
from server.platformcapabilities import HostCapabilities


def make_capabilities(**overrides) -> HostCapabilities:
    data = {
        "system_name": "Linux",
        "system_code": "linux",
        "operating_system_code": "linux",
        "node_name": "test-node",
        "release": "6.8.0",
        "version": "test-version",
        "processor": "x86_64",
        "machine": "x86_64",
        "user_name": "svc-user",
        "user_home_directory": "/srv/service",
        "is_raspberry_pi": False,
        "has_nmcli": False,
        "has_sudo": False,
        "has_timedatectl": False,
        "has_tzutil": False,
        "has_systemsetup": False,
        "has_shutdown": False,
        "has_reboot": False,
        "has_date_command": False,
        "has_ln": False,
        "has_powershell": False,
        "has_lgpio": False,
        "has_periphery": False,
        "has_pinctrl": False,
    }
    data.update(overrides)
    return HostCapabilities(**data)


class OSManagementCapabilityTests(unittest.TestCase):
    def test_set_system_datetime_returns_unsupported_when_linux_commands_are_missing(self):
        capabilities = make_capabilities()

        with (
            mock.patch.object(osmanagement, "get_host_capabilities", return_value=capabilities),
            mock.patch.object(
                osmanagement.subprocess,
                "run",
                side_effect=AssertionError("subprocess.run should not be called"),
            ),
        ):
            success, message = osmanagement.set_system_datetime(
                year=2026,
                month=4,
                day=4,
                hour=12,
                minute=0,
                second=0,
            )

        self.assertFalse(success)
        self.assertIn("unsupported", message.lower())
        self.assertIn("timedatectl", message)

    def test_set_system_timezone_uses_timedatectl_when_available(self):
        capabilities = make_capabilities(has_timedatectl=True)
        completed_process = mock.Mock(returncode=0, stderr="", stdout="")

        with (
            mock.patch.object(osmanagement, "get_host_capabilities", return_value=capabilities),
            mock.patch.object(osmanagement.subprocess, "run", return_value=completed_process) as run_mock,
        ):
            success, message = osmanagement.set_system_timezone(timezone_name="Europe/Prague")

        self.assertTrue(success)
        self.assertEqual(message, "Timezone updated to Europe/Prague")
        run_mock.assert_called_once_with(
            ["timedatectl", "set-timezone", "Europe/Prague"],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_reboot_returns_unsupported_when_linux_reboot_capabilities_are_missing(self):
        capabilities = make_capabilities()

        with (
            mock.patch.object(osmanagement, "get_host_capabilities", return_value=capabilities),
            mock.patch.object(
                osmanagement.subprocess,
                "run",
                side_effect=AssertionError("subprocess.run should not be called"),
            ),
        ):
            success, message = osmanagement.reboot_system()

        self.assertFalse(success)
        self.assertIn("unsupported", message.lower())
        self.assertIn("sudo reboot", message)


if __name__ == "__main__":
    unittest.main()
