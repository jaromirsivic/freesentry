from __future__ import annotations


class SettingsError(RuntimeError):
    """Base class for settings-domain failures."""

    status_code = 500


class SettingsLoadError(SettingsError):
    """Raised when settings cannot be loaded."""


class SettingsSaveError(SettingsError):
    """Raised when settings cannot be persisted."""


class SettingsUpdateError(SettingsError):
    """Raised when a settings mutation cannot be completed."""


class SettingsCacheClearError(SettingsError):
    """Raised when cached settings cannot be invalidated."""
