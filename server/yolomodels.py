from dataclasses import dataclass
from pathlib import Path
import sys
import threading
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - dependency should exist in production
    torch = None

from ultralytics import YOLO

from .ai_setup_constants import (
    ARM_CPU_DEVICE_VALUE,
    CPU_DEVICE_VALUE,
    CUDA_DEVICE_0_VALUE,
    CUDA_DEVICE_1_VALUE,
    DEFAULT_DEVICE,
    DEVICE_OPTIONS,
)

_CUDA_DEVICE_VALUE_BY_TOKEN = {
    "cuda:0": CUDA_DEVICE_0_VALUE,
    "cuda:1": CUDA_DEVICE_1_VALUE,
}


@dataclass(slots=True)
class _ModelCacheEntry:
    model: YOLO
    inference_lock: threading.Lock


@dataclass(slots=True, frozen=True)
class _ResolvedDeviceConfig:
    device_value: str
    model_type: str
    effective_device: str
    warning_message: str | None = None

class YOLOModels:
    """
    Singleton class to manage YOLO models.
    """
    # Model names available for selection (no instance required).
    MODEL_NAMES = [
        "Simple and Fast",
        "Mid. Quality Mid. Speed",
        "Good Quality Moderate Speed",
        "Accurate and Slow",
        "Very Accurate and Very Slow"
    ]
    DEFAULT_MODEL_NAME = "Simple and Fast"

    _instance = None
    _instance_lock = threading.Lock()
    _MODEL_FILENAMES = {
        "pt": {
            MODEL_NAMES[0]: "yolo26n-pose.pt",
            MODEL_NAMES[1]: "yolo26s-pose.pt",
            MODEL_NAMES[2]: "yolo26m-pose.pt",
            MODEL_NAMES[3]: "yolo26l-pose.pt",
            MODEL_NAMES[4]: "yolo26x-pose.pt",
        },
        "ncnn": {
            MODEL_NAMES[0]: "yolo26n-pose_ncnn_model",
            MODEL_NAMES[1]: "yolo26s-pose_ncnn_model",
            MODEL_NAMES[2]: "yolo26m-pose_ncnn_model",
            MODEL_NAMES[3]: "yolo26l-pose_ncnn_model",
            MODEL_NAMES[4]: "yolo26x-pose_ncnn_model",
        },
    }

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super(YOLOModels, cls).__new__(cls)
                    instance._cache_lock = threading.RLock()
                    instance._model_cache: dict[tuple[str, str, str], _ModelCacheEntry] = {}
                    # Keep the most recently used model metadata for compatibility.
                    instance._cached_model_name = None
                    instance._cached_model_type = None
                    instance._cached_model_device = None
                    instance._cached_model = None
                    instance._logged_device_warnings: set[str] = set()
                    cls._instance = instance
        return cls._instance

    @staticmethod
    def _normalize_device_token(device: str | None = DEFAULT_DEVICE) -> str:
        raw_value = str(device or DEFAULT_DEVICE).strip()
        if not raw_value:
            raw_value = DEFAULT_DEVICE
        return (raw_value + " ").split(" ")[0]

    @classmethod
    def _is_cuda_available(cls) -> bool:
        if torch is None:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @classmethod
    def _get_cuda_device_count(cls) -> int:
        if not cls._is_cuda_available():
            return 0
        try:
            return int(torch.cuda.device_count())
        except Exception:
            return 0

    @classmethod
    def _get_cuda_option_value(cls, normalized_device: str) -> str | None:
        return _CUDA_DEVICE_VALUE_BY_TOKEN.get(normalized_device)

    @classmethod
    def _get_device_option_state(cls, option_value: str) -> tuple[bool, str | None]:
        normalized_device = cls._normalize_device_token(option_value)
        if not normalized_device.startswith("cuda"):
            return False, None

        cuda_device_count = cls._get_cuda_device_count()
        if cuda_device_count <= 0:
            return True, "CUDA is not available in the active backend runtime."

        try:
            requested_index = int(normalized_device.split(":", 1)[1])
        except (IndexError, ValueError):
            return True, "Invalid CUDA device identifier."

        if requested_index >= cuda_device_count:
            return (
                True,
                f"Only {cuda_device_count} CUDA device(s) are available in the active backend runtime.",
            )

        return False, None

    @classmethod
    def get_supported_device_options(cls) -> list[dict[str, Any]]:
        supported_options: list[dict[str, Any]] = []
        for option in DEVICE_OPTIONS:
            next_option: dict[str, Any] = dict(option)
            disabled, reason = cls._get_device_option_state(option["value"])
            next_option["disabled"] = disabled
            if reason is not None:
                next_option["reason"] = reason
            supported_options.append(next_option)
        return supported_options

    @classmethod
    def get_default_device_value(cls) -> str:
        supported_options = cls.get_supported_device_options()
        supported_values = {
            option["value"]
            for option in supported_options
            if not bool(option.get("disabled", False))
        }
        if DEFAULT_DEVICE in supported_values:
            return DEFAULT_DEVICE
        if supported_options:
            first_enabled_option = next(
                (
                    option["value"]
                    for option in supported_options
                    if not bool(option.get("disabled", False))
                ),
                None,
            )
            if first_enabled_option is not None:
                return first_enabled_option
        return DEFAULT_DEVICE

    @classmethod
    def _resolve_device_config(
        cls,
        *,
        device: str | None = DEFAULT_DEVICE,
    ) -> _ResolvedDeviceConfig:
        normalized_device = cls._normalize_device_token(device)
        default_device_value = cls.get_default_device_value()

        if normalized_device == "arm_cpu":
            return _ResolvedDeviceConfig(
                device_value=ARM_CPU_DEVICE_VALUE,
                model_type="ncnn",
                effective_device="cpu",
            )

        if normalized_device.startswith("cuda"):
            cuda_device_count = cls._get_cuda_device_count()
            cuda_device_value = cls._get_cuda_option_value(normalized_device)
            if cuda_device_value is None:
                return _ResolvedDeviceConfig(
                    device_value=default_device_value,
                    model_type="pt",
                    effective_device="cpu",
                    warning_message=(
                        f"requested unsupported device '{device}'; "
                        f"falling back to '{default_device_value}'"
                    ),
                )
            if cuda_device_count <= 0:
                return _ResolvedDeviceConfig(
                    device_value=default_device_value,
                    model_type="pt",
                    effective_device="cpu",
                    warning_message=(
                        f"requested '{device}' but CUDA is unavailable in the project runtime; "
                        f"falling back to '{default_device_value}'"
                    ),
                )
            try:
                requested_index = int(normalized_device.split(":", 1)[1])
            except (IndexError, ValueError):
                requested_index = -1
            if 0 <= requested_index < cuda_device_count:
                return _ResolvedDeviceConfig(
                    device_value=cuda_device_value,
                    model_type="pt",
                    effective_device=normalized_device,
                )
            return _ResolvedDeviceConfig(
                device_value=default_device_value,
                model_type="pt",
                effective_device="cpu",
                warning_message=(
                    f"requested '{device}' but only {cuda_device_count} CUDA device(s) are available; "
                    f"falling back to '{default_device_value}'"
                ),
            )

        if normalized_device != "cpu":
            return _ResolvedDeviceConfig(
                device_value=default_device_value,
                model_type="pt",
                effective_device="cpu",
                warning_message=(
                    f"requested unsupported device '{device}'; "
                    f"falling back to '{default_device_value}'"
                ),
            )

        return _ResolvedDeviceConfig(
            device_value=CPU_DEVICE_VALUE,
            model_type="pt",
            effective_device="cpu",
        )

    @classmethod
    def normalize_device_value(cls, device: str | None = DEFAULT_DEVICE) -> str:
        return cls._resolve_device_config(device=device).device_value

    @classmethod
    def get_runtime_diagnostics(cls) -> dict[str, Any]:
        torch_version = getattr(torch, "__version__", None) if torch is not None else None
        torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None) if torch is not None else None
        return {
            "python_executable": sys.executable,
            "torch_module_path": getattr(torch, "__file__", None) if torch is not None else None,
            "torch_version": torch_version,
            "torch_cuda_version": torch_cuda_version,
            "cuda_available": cls._is_cuda_available(),
            "cuda_device_count": cls._get_cuda_device_count(),
            "default_device": cls.get_default_device_value(),
            "device_options": cls.get_supported_device_options(),
        }

    def _log_device_warning_once(self, *, warning_message: str | None) -> None:
        if not warning_message:
            return
        with self._cache_lock:
            if warning_message in self._logged_device_warnings:
                return
            self._logged_device_warnings.add(warning_message)
        print(f"AI inference device fallback: {warning_message}")

    def _resolve_model_filename(self, *, model_name: str, model_type: str) -> tuple[str, str, str]:
        filenames = self._MODEL_FILENAMES.get(model_type)
        if filenames is None:
            print(f"Invalid model type: {model_type}. The only valid types are 'pt' and 'ncnn'.")
            model_type = "pt"
            filenames = self._MODEL_FILENAMES[model_type]
        if model_name not in filenames:
            print(f"Invalid model name: {model_name}")
            model_name = self.DEFAULT_MODEL_NAME
        return model_name, model_type, filenames[model_name]

    def _update_last_model_locked(self, *, cache_key: tuple[str, str, str], model: YOLO) -> None:
        self._cached_model_name, self._cached_model_type, self._cached_model_device = cache_key
        self._cached_model = model

    def _load_model_entry(
        self,
        *,
        model_name: str,
        model_type: str,
        device: str = "cpu",
        model_filename: str | None = None,
    ) -> _ModelCacheEntry | None:
        # path of a directory of this file
        this_file_path = Path(__file__).resolve()
        model_dir = this_file_path.parent / "ai_models" / "yolo"
        # check if the model filename is valid
        if model_filename is None:
            print(f"Model filename is not valid: {model_filename}")
            return None
        model_path = model_dir / model_filename
        # check if the model filename exists
        if not model_path.exists():
            print(f"Model filename does not exist: {model_filename}")
            return None

        model = YOLO(model_path)
        # Exported runtimes such as NCNN do not support PyTorch-style device transfers.
        if model_type == "pt":
            model.to(device=device)
        return _ModelCacheEntry(model=model, inference_lock=threading.Lock())

    def _get_or_load_model_entry(self, *, model_name: str, device: str = "cpu") -> _ModelCacheEntry | None:
        resolved_device = self._resolve_device_config(device=device)
        self._log_device_warning_once(warning_message=resolved_device.warning_message)
        resolved_model_name, resolved_model_type, model_filename = self._resolve_model_filename(
            model_name=model_name,
            model_type=resolved_device.model_type,
        )
        cache_key = (resolved_model_name, resolved_model_type, resolved_device.effective_device)
        with self._cache_lock:
            cached_entry = self._model_cache.get(cache_key)
            if cached_entry is not None:
                self._update_last_model_locked(cache_key=cache_key, model=cached_entry.model)
                return cached_entry

            cached_entry = self._load_model_entry(
                model_name=resolved_model_name,
                model_type=resolved_model_type,
                device=resolved_device.effective_device,
                model_filename=model_filename,
            )
            if cached_entry is None:
                return None

            self._model_cache[cache_key] = cached_entry
            self._update_last_model_locked(cache_key=cache_key, model=cached_entry.model)
            return cached_entry

    def convert_model(self):
        for model_name in YOLOModels.MODEL_NAMES:
            model = self.get_model(model_name=model_name, device="cpu")
            model.export(format="ncnn")
            model.export(format="engine")
            model.export(format="openvino")

    def get_model(self, *,model_name: str, device: str = "cpu") -> YOLO:
        model_entry = self._get_or_load_model_entry(model_name=model_name, device=device)
        if model_entry is None:
            raise RuntimeError(f"Unable to load YOLO model '{model_name}' for device '{device}'")
        return model_entry.model

    def predict(self, *, model_name: str, device: str = "cpu", image: Any, **kwargs) -> Any:
        model_entry = self._get_or_load_model_entry(model_name=model_name, device=device)
        if model_entry is None:
            raise RuntimeError(f"Unable to load YOLO model '{model_name}' for device '{device}'")
        with model_entry.inference_lock:
            return model_entry.model.predict(source=image, **kwargs)

    @property
    def default_model_name(self) -> str:
        return YOLOModels.MODEL_NAMES[0]
