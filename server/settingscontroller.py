from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from fastapi import HTTPException
from pathlib import Path
from tempfile import mkstemp
from typing import Any, TypeVar
import json
import os
import threading

# Get the directory of the current file
BASE_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = BASE_DIR / "wwwroot/src/assets/settings.json"
_SETTINGS_SECTION_TYPES: dict[str, type[Any]] = {
    "general": dict,
    "motors": list,
    "hotZone": dict,
    "aiSetup": dict,
    "manualControl": dict,
    "cameras": dict,
    "primaryCamera": dict,
}
_SUCCESS_RESPONSE = {"success": True, "message": "Settings saved successfully"}
_CLEAR_CACHE_RESPONSE = {"success": True, "message": "Cached settings dropped successfully"}
T = TypeVar("T")


class FrozenList(Sequence[Any]):
    """Read-only list wrapper used by the sync settings fast path."""

    __slots__ = ("_values",)

    def __init__(self, values: Iterable[Any]):
        self._values = tuple(values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __repr__(self) -> str:
        return repr(list(self._values))

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        return [deepcopy(value, memo) for value in self._values]


class FrozenDict(Mapping[str, Any]):
    """Read-only dict wrapper used by the sync settings fast path."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]):
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def items(self):
        return self._values.items()

    def keys(self):
        return self._values.keys()

    def values(self):
        return self._values.values()

    def __repr__(self) -> str:
        return repr(self._values)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return {key: deepcopy(value, memo) for key, value in self._values.items()}


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(_freeze_value(item) for item in value)
    return value


class SettingsStore:
    """Atomic settings store with cached mutable copies and immutable snapshots."""

    def __init__(self, settings_file: Path):
        self._settings_file = settings_file
        self._lock = threading.RLock()
        self._cached_settings: dict[str, Any] | None = None
        self._cached_snapshot: FrozenDict | None = None

    def get_copy(self) -> dict[str, Any]:
        with self._lock:
            settings = self._load_locked()
            return deepcopy(settings)

    def get_snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            self._load_locked()
            if self._cached_snapshot is None:
                raise RuntimeError("Settings snapshot was not initialized")
            return self._cached_snapshot

    def save(self, settings: Mapping[str, Any]) -> dict[str, str | bool]:
        with self._lock:
            normalized = self._validate_and_normalize_settings(settings)
            self._write_locked(normalized)
            self._publish_locked(normalized)
        return dict(_SUCCESS_RESPONSE)

    def update(self, mutator: Callable[[dict[str, Any]], T]) -> T:
        with self._lock:
            current_settings = deepcopy(self._load_locked())
            result = mutator(current_settings)
            normalized = self._validate_and_normalize_settings(current_settings)
            self._write_locked(normalized)
            self._publish_locked(normalized)
            return result

    def clear_cache(self) -> dict[str, str | bool]:
        with self._lock:
            self._cached_settings = None
            self._cached_snapshot = None
        return dict(_CLEAR_CACHE_RESPONSE)

    def _load_locked(self) -> dict[str, Any]:
        if self._cached_settings is None:
            with open(self._settings_file, "r", encoding="utf-8") as settings_file:
                settings = json.load(settings_file)
            normalized = self._validate_and_normalize_settings(settings)
            self._publish_locked(normalized)
        if self._cached_settings is None:
            raise RuntimeError("Settings cache was not initialized")
        return self._cached_settings

    def _publish_locked(self, settings: dict[str, Any]) -> None:
        self._cached_settings = settings
        frozen_settings = _freeze_value(settings)
        if not isinstance(frozen_settings, FrozenDict):
            raise TypeError("Settings root must be a dictionary")
        self._cached_snapshot = frozen_settings

    def _validate_and_normalize_settings(self, settings: Mapping[str, Any]) -> dict[str, Any]:
        normalized = deepcopy(dict(settings))
        for section_name, expected_type in _SETTINGS_SECTION_TYPES.items():
            if section_name not in normalized:
                continue
            section_value = normalized[section_name]
            if not isinstance(section_value, expected_type):
                raise TypeError(
                    f"settings['{section_name}'] must be of type {expected_type.__name__}"
                )
        return normalized

    def _write_locked(self, settings: dict[str, Any]) -> None:
        self._settings_file.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_file_name = mkstemp(
            prefix=f"{self._settings_file.name}.",
            suffix=".tmp",
            dir=self._settings_file.parent,
        )
        temp_path = Path(temp_file_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as temp_file:
                json.dump(settings, temp_file, indent=4, ensure_ascii=False)
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self._settings_file)
            self._fsync_parent_directory()
        except Exception:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

    def _fsync_parent_directory(self) -> None:
        try:
            directory_fd = os.open(self._settings_file.parent, os.O_RDONLY)
        except (AttributeError, OSError, TypeError):
            return
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            os.close(directory_fd)


_SETTINGS_STORE = SettingsStore(SETTINGS_FILE)


def get_settings_sync() -> Mapping[str, Any]:
    """Get a read-only in-memory snapshot of current settings."""
    try:
        return _SETTINGS_STORE.get_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load settings: {str(e)}")


async def get_settings() -> dict[str, Any]:
    """Get a detached mutable copy of current settings."""
    try:
        return _SETTINGS_STORE.get_copy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load settings: {str(e)}")


async def save_settings(settings: dict[str, Any]) -> dict[str, str | bool]:
    """Replace settings.json with the provided settings atomically."""
    try:
        return _SETTINGS_STORE.save(settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


async def update_settings(mutator: Callable[[dict[str, Any]], T]) -> T:
    """Run a serialized read-modify-write transaction over settings.json."""
    try:
        return _SETTINGS_STORE.update(mutator)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")


def clear_cached_settings_sync() -> dict[str, str | bool]:
    """Drop the cached settings snapshot."""
    try:
        return _SETTINGS_STORE.clear_cache()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to drop cached settings: {str(e)}")


async def clear_cached_settings() -> dict[str, str | bool]:
    """Drop the cached settings snapshot."""
    try:
        return _SETTINGS_STORE.clear_cache()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to drop cached settings: {str(e)}")