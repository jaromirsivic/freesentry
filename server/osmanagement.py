"""
OS Management module for system date, time, and timezone operations.
Provides cross-platform support for Windows, Linux, and macOS.
"""
import subprocess
from datetime import datetime, timezone
from typing import Tuple

from .platformcapabilities import HostCapabilities, get_host_capabilities


def get_os_type() -> str:
    """
    Get the current operating system type.
    Returns 'windows', 'linux', or 'darwin' (macOS).
    """
    return get_host_capabilities().system_code


def _run_system_command(
    *,
    command: list[str],
    success_message: str,
    failure_prefix: str,
) -> Tuple[bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "Operation timed out"
    except FileNotFoundError as e:
        return False, f"Required command not found: {e}"
    except Exception as e:
        return False, f"{failure_prefix}: {str(e)}"

    if result.returncode != 0:
        error_output = (result.stderr or result.stdout or "").strip()
        if not error_output:
            error_output = f"command exited with code {result.returncode}"
        return False, f"{failure_prefix}: {error_output}"

    return True, success_message


def _unsupported_message(*, action: str, detail: str) -> Tuple[bool, str]:
    return False, f"{action} is unsupported on this host: {detail}"


def _resolve_datetime_command(
    *,
    capabilities: HostCapabilities,
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
) -> tuple[list[str] | None, str, str | None]:
    os_type = capabilities.system_code

    if os_type == "windows":
        if not capabilities.has_powershell:
            return None, "", "powershell is unavailable"
        datetime_str = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
        return (
            ["powershell", "-Command", f'Set-Date -Date "{datetime_str}"'],
            "Date and time updated successfully",
            None,
        )

    if os_type == "linux":
        datetime_str = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
        if capabilities.has_timedatectl:
            return (
                ["timedatectl", "set-time", datetime_str],
                "Date and time updated successfully",
                None,
            )
        if capabilities.has_sudo and capabilities.has_date_command:
            date_str = f"{month:02d}{day:02d}{hour:02d}{minute:02d}{year:04d}.{second:02d}"
            return (
                ["sudo", "date", date_str],
                "Date and time updated successfully",
                None,
            )
        return None, "", "neither timedatectl nor sudo date is available"

    if os_type == "darwin":
        if not capabilities.has_sudo or not capabilities.has_date_command:
            return None, "", "sudo date is unavailable"
        datetime_str = f"{month:02d}{day:02d}{hour:02d}{minute:02d}{year:04d}.{second:02d}"
        return (
            ["sudo", "date", datetime_str],
            "Date and time updated successfully",
            None,
        )

    return None, "", f"unsupported operating system: {os_type}"


def _resolve_timezone_command(
    *,
    capabilities: HostCapabilities,
    timezone_name: str,
) -> tuple[list[str] | None, str, str | None]:
    os_type = capabilities.system_code

    if os_type == "windows":
        if not capabilities.has_tzutil:
            return None, "", "tzutil is unavailable"
        windows_tz_map = {
            "UTC": "UTC",
            "America/New_York": "Eastern Standard Time",
            "America/Chicago": "Central Standard Time",
            "America/Denver": "Mountain Standard Time",
            "America/Los_Angeles": "Pacific Standard Time",
            "America/Anchorage": "Alaskan Standard Time",
            "Pacific/Honolulu": "Hawaiian Standard Time",
            "Europe/London": "GMT Standard Time",
            "Europe/Paris": "Romance Standard Time",
            "Europe/Berlin": "W. Europe Standard Time",
            "Europe/Moscow": "Russian Standard Time",
            "Asia/Tokyo": "Tokyo Standard Time",
            "Asia/Shanghai": "China Standard Time",
            "Asia/Singapore": "Singapore Standard Time",
            "Asia/Dubai": "Arabian Standard Time",
            "Asia/Kolkata": "India Standard Time",
            "Australia/Sydney": "AUS Eastern Standard Time",
            "Australia/Perth": "W. Australia Standard Time",
            "Pacific/Auckland": "New Zealand Standard Time"
        }
        windows_tz = windows_tz_map.get(timezone_name, timezone_name)
        return (
            ["tzutil", "/s", windows_tz],
            f"Timezone updated to {timezone_name}",
            None,
        )

    if os_type == "linux":
        if capabilities.has_timedatectl:
            return (
                ["timedatectl", "set-timezone", timezone_name],
                f"Timezone updated to {timezone_name}",
                None,
            )

        if capabilities.has_sudo and capabilities.has_ln:
            import os

            tz_file = f"/usr/share/zoneinfo/{timezone_name}"
            if not os.path.exists(tz_file):
                return None, "", f"timezone file not found: {tz_file}"
            return (
                ["sudo", "ln", "-sf", tz_file, "/etc/localtime"],
                f"Timezone updated to {timezone_name}",
                None,
            )

        return None, "", "neither timedatectl nor sudo ln is available"

    if os_type == "darwin":
        if not capabilities.has_sudo or not capabilities.has_systemsetup:
            return None, "", "sudo systemsetup is unavailable"
        return (
            ["sudo", "systemsetup", "-settimezone", timezone_name],
            f"Timezone updated to {timezone_name}",
            None,
        )

    return None, "", f"unsupported operating system: {os_type}"


def _resolve_reboot_command(
    *,
    capabilities: HostCapabilities,
) -> tuple[list[str] | None, str, str | None]:
    os_type = capabilities.system_code

    if os_type == "windows":
        if not capabilities.has_shutdown:
            return None, "", "shutdown is unavailable"
        return (
            ["shutdown", "/r", "/t", "5", "/c", "Reboot requested"],
            "System reboot initiated",
            None,
        )

    if os_type in {"linux", "darwin"}:
        if not capabilities.has_sudo or not capabilities.has_reboot:
            return None, "", "sudo reboot is unavailable"
        return (
            ["sudo", "reboot"],
            "System reboot initiated",
            None,
        )

    return None, "", f"unsupported operating system: {os_type}"


def get_current_datetime() -> datetime:
    """Get the current system datetime."""
    return datetime.now()


def get_current_datetime_utc() -> datetime:
    """Get the current system datetime in UTC."""
    return datetime.now(timezone.utc)


def get_timezone_name() -> str:
    """Get the current system timezone name."""
    return datetime.now().astimezone().strftime('%Z')


def get_timezone_offset_minutes() -> int:
    """
    Get the current timezone offset in minutes from UTC.
    Positive values mean ahead of UTC (e.g., +120 for UTC+2).
    """
    now = datetime.now()
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    offset_seconds = (now - utc_now).total_seconds()
    return int(offset_seconds / 60)


def get_available_timezones() -> list[str]:
    """
    Get list of available timezone names.
    Returns a list of common timezone identifiers.
    """
    # Common timezones that work across platforms
    return [
        "UTC",
        "America/New_York",
        "America/Chicago", 
        "America/Denver",
        "America/Los_Angeles",
        "America/Anchorage",
        "Pacific/Honolulu",
        "Europe/London",
        "Europe/Paris",
        "Europe/Berlin",
        "Europe/Moscow",
        "Asia/Tokyo",
        "Asia/Shanghai",
        "Asia/Singapore",
        "Asia/Dubai",
        "Asia/Kolkata",
        "Australia/Sydney",
        "Australia/Perth",
        "Pacific/Auckland"
    ]


def set_system_datetime(*, year: int, month: int, day: int, 
                        hour: int, minute: int, second: int) -> Tuple[bool, str]:
    """
    Set the system date and time.
    Requires elevated privileges on all platforms.
    
    Args:
        year: Year (e.g., 2024)
        month: Month (1-12)
        day: Day of month (1-31)
        hour: Hour (0-23)
        minute: Minute (0-59)
        second: Second (0-59)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    capabilities = get_host_capabilities()
    command, success_message, unsupported_detail = _resolve_datetime_command(
        capabilities=capabilities,
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
    )

    if command is None:
        return _unsupported_message(action="Setting system date/time", detail=unsupported_detail or "no valid command path")

    return _run_system_command(
        command=command,
        success_message=success_message,
        failure_prefix="Failed to set date/time",
    )


def set_system_timezone(*, timezone_name: str) -> Tuple[bool, str]:
    """
    Set the system timezone.
    Requires elevated privileges on all platforms.
    
    Args:
        timezone_name: Timezone identifier (e.g., 'America/New_York', 'Europe/London')
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    capabilities = get_host_capabilities()
    command, success_message, unsupported_detail = _resolve_timezone_command(
        capabilities=capabilities,
        timezone_name=timezone_name,
    )

    if command is None:
        return _unsupported_message(action="Setting system timezone", detail=unsupported_detail or "no valid command path")

    return _run_system_command(
        command=command,
        success_message=success_message,
        failure_prefix="Failed to set timezone",
    )


def reboot_system() -> Tuple[bool, str]:
    """
    Reboot the system.
    Requires elevated privileges on all platforms.

    Returns:
        Tuple of (success: bool, message: str)
    """
    capabilities = get_host_capabilities()
    command, success_message, unsupported_detail = _resolve_reboot_command(capabilities=capabilities)

    if command is None:
        return _unsupported_message(action="Rebooting the system", detail=unsupported_detail or "no valid command path")

    return _run_system_command(
        command=command,
        success_message=success_message,
        failure_prefix="Failed to initiate reboot",
    )


def get_system_info() -> dict:
    """
    Get current system date/time information.
    
    Returns:
        Dictionary with datetime, timezone, and offset information
    """
    now = get_current_datetime()
    utc_now = get_current_datetime_utc()
    
    return {
        "dateTime": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "dateTimeUtc": utc_now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timezoneName": get_timezone_name(),
        "timezoneOffsetMinutes": get_timezone_offset_minutes(),
        "osType": get_os_type()
    }
