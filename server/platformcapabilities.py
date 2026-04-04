from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import getpass
import importlib.util
import os
from pathlib import Path
import platform
import shutil


def _read_text_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
    except OSError:
        return ""


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return False


def _command_available(command_name: str) -> bool:
    return shutil.which(command_name) is not None


def _safe_get_user_name() -> str:
    try:
        user_name = getpass.getuser()
        if user_name:
            return user_name
    except Exception:
        pass

    for env_name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value

    try:
        return str(os.getuid())
    except (AttributeError, OSError):
        return "unknown"


def _is_raspberry_pi(*, system_code: str, uname_info: platform.uname_result) -> bool:
    if system_code != "linux":
        return False

    detection_hints = " ".join(
        value
        for value in (
            _read_text_file("/proc/device-tree/model"),
            _read_text_file("/sys/firmware/devicetree/base/model"),
            uname_info.node,
            uname_info.machine,
            uname_info.processor,
            uname_info.version,
        )
        if value
    ).lower()

    return (
        "raspberry pi" in detection_hints
        or "raspberrypi" in detection_hints
        or detection_hints.startswith("rpi ")
    )


def _normalize_operating_system_code(*, system_code: str, is_raspberry_pi: bool) -> str:
    if system_code == "darwin":
        return "macos"
    if system_code == "linux" and is_raspberry_pi:
        return "raspberrypi5"
    return system_code or "unknown"


@dataclass(frozen=True)
class HostCapabilities:
    system_name: str
    system_code: str
    operating_system_code: str
    node_name: str
    release: str
    version: str
    processor: str
    machine: str
    user_name: str
    user_home_directory: str
    is_raspberry_pi: bool
    has_nmcli: bool
    has_sudo: bool
    has_timedatectl: bool
    has_tzutil: bool
    has_systemsetup: bool
    has_shutdown: bool
    has_reboot: bool
    has_date_command: bool
    has_ln: bool
    has_powershell: bool
    has_lgpio: bool
    has_periphery: bool
    has_pinctrl: bool

    @property
    def startup_script_os_codes(self) -> tuple[str, ...]:
        if self.operating_system_code == "macos":
            return ("macos", "darwin")
        return (self.operating_system_code,)

    @property
    def supports_wifi_configuration(self) -> bool:
        return self.operating_system_code.startswith("raspberrypi") and self.has_sudo and self.has_nmcli

    @property
    def supports_gpio(self) -> bool:
        return self.is_raspberry_pi and self.has_lgpio

    @property
    def supports_hardware_pwm(self) -> bool:
        return self.is_raspberry_pi and self.has_periphery and self.has_pinctrl


@lru_cache(maxsize=1)
def get_host_capabilities() -> HostCapabilities:
    uname_info = platform.uname()
    system_code = uname_info.system.lower()
    is_raspberry_pi = _is_raspberry_pi(system_code=system_code, uname_info=uname_info)

    return HostCapabilities(
        system_name=uname_info.system,
        system_code=system_code,
        operating_system_code=_normalize_operating_system_code(
            system_code=system_code,
            is_raspberry_pi=is_raspberry_pi,
        ),
        node_name=uname_info.node,
        release=uname_info.release,
        version=uname_info.version,
        processor=uname_info.processor,
        machine=uname_info.machine,
        user_name=_safe_get_user_name(),
        user_home_directory=os.path.expanduser("~"),
        is_raspberry_pi=is_raspberry_pi,
        has_nmcli=_command_available("nmcli"),
        has_sudo=_command_available("sudo"),
        has_timedatectl=_command_available("timedatectl"),
        has_tzutil=_command_available("tzutil"),
        has_systemsetup=_command_available("systemsetup"),
        has_shutdown=_command_available("shutdown"),
        has_reboot=_command_available("reboot"),
        has_date_command=_command_available("date"),
        has_ln=_command_available("ln"),
        has_powershell=_command_available("powershell"),
        has_lgpio=_module_available("gpiozero.pins.lgpio"),
        has_periphery=_module_available("periphery"),
        has_pinctrl=_command_available("pinctrl"),
    )


def clear_host_capabilities_cache() -> None:
    get_host_capabilities.cache_clear()
