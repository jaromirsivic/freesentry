from __future__ import annotations

from datetime import datetime


def _normalize_candidate(value: str, *, field_name: str) -> str:
    candidate = value.strip()
    if len(candidate) == 0:
        raise ValueError(f"{field_name} must be a non-empty ISO 8601 datetime string")
    if candidate.endswith(("Z", "z")):
        candidate = f"{candidate[:-1]}+00:00"
    return candidate


def coerce_iso_datetime(value: str, *, field_name: str = "datetime") -> tuple[str, datetime]:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    candidate = _normalize_candidate(value, field_name=field_name)
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO 8601 datetime string") from exc

    return parsed.isoformat(), parsed


def normalize_iso_datetime_string(value: str, *, field_name: str = "datetime") -> str:
    normalized, _ = coerce_iso_datetime(value, field_name=field_name)
    return normalized


def compare_datetimes(left: datetime, right: datetime) -> int:
    if left.tzinfo is None or right.tzinfo is None:
        left = left.replace(tzinfo=None)
        right = right.replace(tzinfo=None)

    if left < right:
        return -1
    if left > right:
        return 1
    return 0
