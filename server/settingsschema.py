from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, field_validator

from .isodatetime import normalize_iso_datetime_string


class HistogramPointModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    pwmMultiplier: int | float
    forwardSeconds: int | float
    reverseSeconds: int | float

    @field_validator("pwmMultiplier")
    @classmethod
    def validate_pwm_multiplier(cls, value: int | float) -> int | float:
        if isinstance(value, bool) or value < 0 or value > 1:
            raise ValueError("pwmMultiplier must be in range [0, 1]")
        return value

    @field_validator("forwardSeconds", "reverseSeconds")
    @classmethod
    def validate_duration(cls, value: int | float) -> int | float:
        if isinstance(value, bool) or value < 0:
            raise ValueError("duration must be non-negative")
        return value


class ExitStrategyConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    fixedDateTime: str | None = None

    @field_validator("fixedDateTime")
    @classmethod
    def normalize_fixed_datetime(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_iso_datetime_string(
            value,
            field_name="settings['aiSetup']['exitStrategy']['fixedDateTime']",
        )


class AISetupConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    activationDateTime: str | None = None
    exitStrategy: ExitStrategyConfigModel | None = None

    @field_validator("activationDateTime")
    @classmethod
    def normalize_activation_datetime(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_iso_datetime_string(
            value,
            field_name="settings['aiSetup']['activationDateTime']",
        )


_HISTOGRAM_ADAPTER = TypeAdapter(list[HistogramPointModel])


def normalize_settings(settings: dict[str, Any]) -> dict[str, Any]:
    motors = settings.get("motors")
    if isinstance(motors, list):
        settings["motors"] = [
            _normalize_motor_settings(motor, motor_index=motor_index)
            for motor_index, motor in enumerate(motors)
        ]

    ai_setup = settings.get("aiSetup")
    if isinstance(ai_setup, dict):
        settings["aiSetup"] = _normalize_ai_setup(ai_setup)

    return settings


def _normalize_motor_settings(motor: Any, *, motor_index: int) -> dict[str, Any]:
    if not isinstance(motor, Mapping):
        raise TypeError(f"settings['motors'][{motor_index}] must be a dictionary")

    normalized_motor = dict(motor)
    if "histogram" not in normalized_motor:
        return normalized_motor

    field_name = f"settings['motors'][{motor_index}]['histogram']"
    histogram = normalized_motor["histogram"]
    try:
        histogram_models = _HISTOGRAM_ADAPTER.validate_python(histogram)
    except ValidationError as exc:
        raise ValueError(f"{field_name} {_summarize_validation_error(exc)}") from exc

    _validate_histogram_invariants(histogram_models, field_name=field_name)
    normalized_motor["histogram"] = [
        histogram_model.model_dump(mode="python")
        for histogram_model in histogram_models
    ]
    return normalized_motor


def _normalize_ai_setup(ai_setup: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized_ai_setup = AISetupConfigModel.model_validate(ai_setup)
    except ValidationError as exc:
        raise ValueError(f"settings['aiSetup'] {_summarize_validation_error(exc)}") from exc

    return normalized_ai_setup.model_dump(mode="python", exclude_none=True)


def _validate_histogram_invariants(
    histogram_points: list[HistogramPointModel],
    *,
    field_name: str,
) -> None:
    if len(histogram_points) < 2:
        raise ValueError(f"{field_name} must contain at least two items")

    sorted_points = sorted(histogram_points, key=lambda point: float(point.pwmMultiplier))
    first_point = sorted_points[0]
    last_point = sorted_points[-1]

    if (
        float(first_point.pwmMultiplier) != 0
        or float(first_point.forwardSeconds) != 0
        or float(first_point.reverseSeconds) != 0
    ):
        raise ValueError(
            f"first item in {field_name} must have pwmMultiplier 0 "
            "and forwardSeconds 0 and reverseSeconds 0"
        )

    if float(last_point.pwmMultiplier) != 1:
        raise ValueError(f"last item in {field_name} must have pwmMultiplier 1")

    if float(last_point.forwardSeconds) <= 0:
        raise ValueError(f"last item in {field_name} must have forwardSeconds greater than 0")

    if float(last_point.reverseSeconds) <= 0:
        raise ValueError(f"last item in {field_name} must have reverseSeconds greater than 0")

    for previous_point, current_point in zip(sorted_points, sorted_points[1:]):
        previous_forward = float(previous_point.forwardSeconds)
        current_forward = float(current_point.forwardSeconds)
        if previous_forward > 0 and current_forward >= previous_forward:
            raise ValueError(f"forwardSeconds in {field_name} must decrease after the first positive point")

        previous_reverse = float(previous_point.reverseSeconds)
        current_reverse = float(current_point.reverseSeconds)
        if previous_reverse > 0 and current_reverse >= previous_reverse:
            raise ValueError(f"reverseSeconds in {field_name} must decrease after the first positive point")


def _summarize_validation_error(exc: ValidationError) -> str:
    first_error = exc.errors(include_url=False)[0]
    location = "".join(f"[{item}]" if isinstance(item, int) else f".{item}" for item in first_error["loc"])
    message = first_error["msg"]
    if location.startswith("."):
        location = location[1:]
    if len(location) == 0:
        return message
    return f"{location}: {message}"
