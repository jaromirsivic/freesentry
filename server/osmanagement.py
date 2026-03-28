"""
OS Management module for system date, time, and timezone operations.
Provides cross-platform support for Windows, Linux, and macOS.
"""
import platform
import subprocess
from datetime import datetime, timezone
from typing import Tuple


def get_os_type() -> str:
    """
    Get the current operating system type.
    Returns 'windows', 'linux', or 'darwin' (macOS).
    """
    return platform.system().lower()


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
    os_type = get_os_type()
    
    try:
        if os_type == 'windows':
            # Windows: Use PowerShell to set date/time
            datetime_str = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
            cmd = ['powershell', '-Command', f'Set-Date -Date "{datetime_str}"']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return False, f"Failed to set date/time: {result.stderr}"
            return True, "Date and time updated successfully"
            
        elif os_type == 'linux':
            # Linux: Use timedatectl or date command
            datetime_str = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
            
            # Try timedatectl first (systemd)
            cmd = ['timedatectl', 'set-time', datetime_str]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                # Fall back to date command
                date_str = f"{month:02d}{day:02d}{hour:02d}{minute:02d}{year:04d}.{second:02d}"
                cmd = ['sudo', 'date', date_str]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    return False, f"Failed to set date/time: {result.stderr}"
            
            return True, "Date and time updated successfully"
            
        elif os_type == 'darwin':
            # macOS: Use date command with specific format
            # Format: [[mm]dd]HH]MM[[cc]yy][.ss]
            datetime_str = f"{month:02d}{day:02d}{hour:02d}{minute:02d}{year:04d}.{second:02d}"
            cmd = ['sudo', 'date', datetime_str]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return False, f"Failed to set date/time: {result.stderr}"
            return True, "Date and time updated successfully"
            
        else:
            return False, f"Unsupported operating system: {os_type}"
            
    except subprocess.TimeoutExpired:
        return False, "Operation timed out"
    except FileNotFoundError as e:
        return False, f"Required command not found: {e}"
    except Exception as e:
        return False, f"Error setting date/time: {str(e)}"


def set_system_timezone(*, timezone_name: str) -> Tuple[bool, str]:
    """
    Set the system timezone.
    Requires elevated privileges on all platforms.
    
    Args:
        timezone_name: Timezone identifier (e.g., 'America/New_York', 'Europe/London')
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    os_type = get_os_type()
    
    try:
        if os_type == 'windows':
            # Windows: Use tzutil to set timezone
            # Windows uses different timezone names, need to map them
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
            cmd = ['tzutil', '/s', windows_tz]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return False, f"Failed to set timezone: {result.stderr}"
            return True, f"Timezone updated to {timezone_name}"
            
        elif os_type == 'linux':
            # Linux: Use timedatectl to set timezone
            cmd = ['timedatectl', 'set-timezone', timezone_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                # Try alternative method using symlink
                try:
                    import os
                    tz_file = f"/usr/share/zoneinfo/{timezone_name}"
                    if os.path.exists(tz_file):
                        cmd = ['sudo', 'ln', '-sf', tz_file, '/etc/localtime']
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        if result.returncode != 0:
                            return False, f"Failed to set timezone: {result.stderr}"
                    else:
                        return False, f"Timezone file not found: {tz_file}"
                except Exception as e:
                    return False, f"Failed to set timezone: {str(e)}"
            
            return True, f"Timezone updated to {timezone_name}"
            
        elif os_type == 'darwin':
            # macOS: Use systemsetup command
            cmd = ['sudo', 'systemsetup', '-settimezone', timezone_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return False, f"Failed to set timezone: {result.stderr}"
            return True, f"Timezone updated to {timezone_name}"
            
        else:
            return False, f"Unsupported operating system: {os_type}"
            
    except subprocess.TimeoutExpired:
        return False, "Operation timed out"
    except FileNotFoundError as e:
        return False, f"Required command not found: {e}"
    except Exception as e:
        return False, f"Error setting timezone: {str(e)}"


def reboot_system() -> Tuple[bool, str]:
    """
    Reboot the system.
    Requires elevated privileges on all platforms.

    Returns:
        Tuple of (success: bool, message: str)
    """
    os_type = get_os_type()

    try:
        if os_type == 'windows':
            cmd = ['shutdown', '/r', '/t', '5', '/c', 'Reboot requested']
        elif os_type == 'linux':
            cmd = ['sudo', 'reboot']
        elif os_type == 'darwin':
            cmd = ['sudo', 'reboot']
        else:
            return False, f"Unsupported operating system: {os_type}"

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return False, f"Failed to initiate reboot: {result.stderr}"
        return True, "System reboot initiated"

    except subprocess.TimeoutExpired:
        return False, "Operation timed out"
    except FileNotFoundError as e:
        return False, f"Required command not found: {e}"
    except Exception as e:
        return False, f"Error initiating reboot: {str(e)}"


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
