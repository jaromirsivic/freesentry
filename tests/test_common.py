import types
import unittest
from unittest import mock

from server import common, j8
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


class PlatformInfoTests(unittest.TestCase):
    def test_get_platform_info_preserves_public_contract_with_normalized_os_code(self):
        capabilities = make_capabilities(
            system_name="Darwin",
            system_code="darwin",
            operating_system_code="macos",
            node_name="mac-host",
            release="23.4.0",
            version="Darwin Kernel Version",
            processor="arm64",
            machine="arm64",
            user_name="launchd-user",
            user_home_directory="/Users/launchd-user",
        )
        fake_memory = types.SimpleNamespace(
            total=8 * 1024 * 1024 * 1024,
            available=6 * 1024 * 1024 * 1024,
        )

        with (
            mock.patch.object(common, "get_host_capabilities", return_value=capabilities),
            mock.patch.object(common.psutil, "virtual_memory", return_value=fake_memory),
            mock.patch.object(common.psutil, "cpu_percent", return_value=12.5),
        ):
            info = common.get_platform_info()

        self.assertEqual(info["operating_system_code"], "macos")
        self.assertEqual(info["operating_system"], "Darwin 23.4.0 (Darwin Kernel Version)")
        self.assertEqual(info["architecture"], "arm64 (arm64)")
        self.assertEqual(info["user"], "mac-host/launchd-user (HomeDir: /Users/launchd-user)")
        self.assertEqual(info["cpu"], "12.50%")
        self.assertIn("GB (used)", info["ram"])


class J8CapabilityFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        j8.J8._instance = None

    def tearDown(self) -> None:
        j8.J8._instance = None

    def test_j8_uses_dummy_pins_on_non_raspberry_pi_hosts(self):
        capabilities = make_capabilities(
            system_name="Windows",
            system_code="windows",
            operating_system_code="windows",
        )

        with (
            mock.patch.object(j8, "get_host_capabilities", return_value=capabilities),
            mock.patch.object(j8, "get_settings_sync", return_value={"general": {"controllerSetup": {}}}),
        ):
            controller = j8.J8()

        self.assertFalse(controller.initialized)
        self.assertIn("not a Raspberry Pi", controller.error_message)
        self.assertEqual(controller[5].name, "Dummy GPIO 5")


if __name__ == "__main__":
    unittest.main()
