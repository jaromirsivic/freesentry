"""Thermal guard: pause training while the GPU (or CPU) is too hot.

Why: a long run on a laptop GPU can push the package to its thermal limit; the
driver then throttles clocks (slow, but survivable) and in the worst case the
machine resets, which costs everything since the last checkpoint. Pausing the
optimizer loop for a minute whenever the sensors cross a threshold is far
cheaper than a reboot, so the guard is **on by default**.

How it works:

* A daemon thread samples the sensors every ``interval_s`` and applies a
  hysteresis: the guard becomes *hot* as soon as any pause threshold is reached
  (GPU >= ``gpu_pause_c``, CPU >= ``cpu_pause_c``, or the NVIDIA "HW thermal
  slowdown" flag if ``pause_on_hw_slowdown``), and goes back to *cool* only once
  **every** active sensor is at/below its resume threshold and the slowdown
  flag is inactive again.
* The trainer checks :attr:`ThermalGuard.hot` once per optimizer step (an
  ``Event.is_set()``, i.e. free) and calls :meth:`wait_until_cool` which
  sleeps in ``interval_s`` chunks until the thread clears the flag. A pause
  longer than ``max_pause_min`` only logs a warning every ``max_pause_min``;
  the guard never aborts a run.
* Sensors: GPU temperature + throttle flags from ``pynvml`` when it happens to
  be importable, otherwise from ``nvidia-smi --query-gpu=...`` (one ~0.1 s
  subprocess per sample). CPU temperature is best effort: ``psutil
  .sensors_temperatures()`` where it exists (Linux), on Windows the WMI class
  ``MSAcpi_ThermalZoneTemperature`` via PowerShell, probed once at start-up and
  disabled when the firmware does not expose it (most laptops, including the
  development machine, answer "Not supported"). The sensor is a plain callable
  returning a dict (``gpu_temp_c``, ``cpu_temp_c``, ``hw_slowdown``,
  ``sw_slowdown``), so tests inject a fake one.
* No sensor and/or a CPU device → the guard logs that it is inactive and never
  pauses; the trainer code path is identical either way.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..utils.logging import get_logger

logger = get_logger(__name__)

SensorFn = Callable[[], dict[str, Any]]

_NVSMI_QUERY = (
    "--query-gpu=temperature.gpu,clocks_throttle_reasons.hw_thermal_slowdown,"
    "clocks_throttle_reasons.sw_thermal_slowdown"
)
_WMI_COMMAND = (
    "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature "
    "| Select-Object -ExpandProperty CurrentTemperature"
)
_SUBPROCESS_TIMEOUT_S = 15.0


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class ThermalConfig:
    """``training.thermal`` block (defaults = configs/u.json)."""

    enabled: bool = True
    interval_s: float = 10.0
    gpu_pause_c: float = 85.0
    gpu_resume_c: float = 75.0
    cpu_pause_c: float = 95.0
    cpu_resume_c: float = 85.0
    pause_on_hw_slowdown: bool = True
    max_pause_min: float = 60.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ThermalConfig":
        raw = dict(raw or {})
        cfg = cls(
            enabled=bool(raw.get("enabled", cls.enabled)),
            interval_s=float(raw.get("interval_s", cls.interval_s)),
            gpu_pause_c=float(raw.get("gpu_pause_c", cls.gpu_pause_c)),
            gpu_resume_c=float(raw.get("gpu_resume_c", cls.gpu_resume_c)),
            cpu_pause_c=float(raw.get("cpu_pause_c", cls.cpu_pause_c)),
            cpu_resume_c=float(raw.get("cpu_resume_c", cls.cpu_resume_c)),
            pause_on_hw_slowdown=bool(raw.get("pause_on_hw_slowdown", cls.pause_on_hw_slowdown)),
            max_pause_min=float(raw.get("max_pause_min", cls.max_pause_min)),
        )
        if cfg.interval_s <= 0:
            raise ValueError("training.thermal.interval_s must be > 0")
        if cfg.gpu_resume_c > cfg.gpu_pause_c or cfg.cpu_resume_c > cfg.cpu_pause_c:
            raise ValueError("training.thermal: resume thresholds must not exceed the pause thresholds")
        return cfg


@dataclass(frozen=True)
class ThermalSample:
    """One sensor reading; ``None`` = sensor not available / failed this time."""

    gpu_temp_c: float | None = None
    cpu_temp_c: float | None = None
    hw_slowdown: bool = False
    sw_slowdown: bool = False
    time: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, now: float) -> "ThermalSample":
        def _temp(key: str) -> float | None:
            value = raw.get(key)
            if value is None:
                return None
            value = float(value)
            return value if value == value else None  # NaN -> missing

        return cls(
            gpu_temp_c=_temp("gpu_temp_c"),
            cpu_temp_c=_temp("cpu_temp_c"),
            hw_slowdown=bool(raw.get("hw_slowdown", False)),
            sw_slowdown=bool(raw.get("sw_slowdown", False)),
            time=now,
        )


# --------------------------------------------------------------------------- sensors


def _nvidia_smi_path() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if sys.platform == "win32":
        for candidate in (r"C:\Windows\System32\nvidia-smi.exe",
                          r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
            if os.path.isfile(candidate):
                return candidate
    return None


def _subprocess_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_S)
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kwargs


def read_nvidia_smi(executable: str | None = None) -> dict[str, Any]:
    """GPU temperature + thermal throttle flags via ``nvidia-smi`` (max / OR over all GPUs).

    Output looks like ``41, Not Active, Not Active`` (one line per GPU).
    """

    executable = executable or _nvidia_smi_path()
    if executable is None:
        raise RuntimeError("nvidia-smi not found")
    result = subprocess.run([executable, _NVSMI_QUERY, "--format=csv,noheader,nounits"], **_subprocess_kwargs())
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi exit {result.returncode}: {result.stderr.strip()[:200]}")
    temps: list[float] = []
    hw = sw = False
    for line in result.stdout.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 3 or not fields[0]:
            continue
        try:
            temps.append(float(fields[0]))
        except ValueError:
            continue
        hw = hw or fields[1].lower() == "active"
        sw = sw or fields[2].lower() == "active"
    if not temps:
        raise RuntimeError(f"nvidia-smi returned no temperature: {result.stdout.strip()[:200]!r}")
    return {"gpu_temp_c": max(temps), "hw_slowdown": hw, "sw_slowdown": sw}


def _make_pynvml_reader() -> SensorFn | None:
    """Prefer pynvml when importable (no subprocess); never a hard dependency."""

    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - not installed / broken driver binding
        return None
    try:
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        if not handles:
            return None
        hw_mask = int(pynvml.nvmlClocksThrottleReasonHwThermalSlowdown)
        sw_mask = int(pynvml.nvmlClocksThrottleReasonSwThermalSlowdown)
    except Exception:  # noqa: BLE001
        return None

    def read() -> dict[str, Any]:
        temps = [float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)) for h in handles]
        reasons = 0
        for h in handles:
            try:
                reasons |= int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h))
            except Exception:  # noqa: BLE001 - flags are optional, the temperature is what matters
                pass
        return {"gpu_temp_c": max(temps), "hw_slowdown": bool(reasons & hw_mask), "sw_slowdown": bool(reasons & sw_mask)}

    return read


_CPU_CHIP_HINTS = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "cpu-thermal", "acpitz", "soc_thermal")


def read_psutil_cpu() -> float | None:
    """Best-effort package temperature from ``psutil.sensors_temperatures()`` (Linux)."""

    import psutil

    if not hasattr(psutil, "sensors_temperatures"):
        return None
    readings = psutil.sensors_temperatures() or {}
    if not readings:
        return None
    preferred = [k for k in readings if any(h in k.lower() for h in _CPU_CHIP_HINTS)]
    keys = preferred or list(readings)
    temps = [float(e.current) for k in keys for e in readings[k] if e.current is not None]
    return max(temps) if temps else None


def read_wmi_cpu() -> float | None:
    """Windows ACPI thermal zone (tenths of kelvin) via PowerShell; raises when unsupported."""

    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", _WMI_COMMAND], **_subprocess_kwargs()
    )
    if result.returncode != 0:
        raise RuntimeError(f"WMI thermal zone unavailable: {result.stderr.strip()[:200]}")
    values: list[float] = []
    for line in result.stdout.split():
        try:
            values.append(float(line) / 10.0 - 273.15)
        except ValueError:
            continue
    if not values:
        raise RuntimeError("WMI thermal zone returned no values")
    return max(values)


def build_sensor(*, want_gpu: bool, want_cpu: bool = True) -> tuple[SensorFn | None, list[str]]:
    """Probe the sensors once and return ``(reader, active_sensor_names)``.

    Each sub-sensor is tried once here; failures disable it for the whole run so
    the sampling thread never spams a broken backend.
    """

    gpu_reader: SensorFn | None = None
    cpu_reader: Callable[[], float | None] | None = None
    names: list[str] = []

    if want_gpu:
        gpu_reader = _make_pynvml_reader()
        if gpu_reader is not None:
            names.append("gpu:pynvml")
        else:
            exe = _nvidia_smi_path()
            if exe is not None:
                try:
                    read_nvidia_smi(exe)
                    gpu_reader = lambda: read_nvidia_smi(exe)  # noqa: E731
                    names.append("gpu:nvidia-smi")
                except Exception as exc:  # noqa: BLE001
                    logger.info("thermal: nvidia-smi probe failed (%s); GPU sensor disabled", exc)
            else:
                logger.info("thermal: nvidia-smi not found; GPU sensor disabled")

    if want_cpu:
        try:
            if read_psutil_cpu() is not None:
                cpu_reader = read_psutil_cpu
                names.append("cpu:psutil")
        except Exception:  # noqa: BLE001 - psutil raises on some platforms
            cpu_reader = None
        if cpu_reader is None and sys.platform == "win32":
            try:
                if read_wmi_cpu() is not None:
                    cpu_reader = read_wmi_cpu
                    names.append("cpu:wmi")
            except Exception as exc:  # noqa: BLE001
                logger.info("thermal: Windows thermal zone unavailable (%s); CPU sensor disabled", str(exc).splitlines()[0][:120])

    if gpu_reader is None and cpu_reader is None:
        return None, names

    def read() -> dict[str, Any]:
        out: dict[str, Any] = {}
        if gpu_reader is not None:
            out.update(gpu_reader())
        if cpu_reader is not None:
            out["cpu_temp_c"] = cpu_reader()
        return out

    return read, names


# --------------------------------------------------------------------------- guard


class ThermalGuard:
    """Background sensor sampling + hysteresis flag + blocking wait for the trainer.

    ``sensor`` overrides the auto-detected sensors (tests, exotic hardware);
    with ``sensor=None`` the guard probes the machine itself and is inactive
    on CPU devices. ``clock``/``sleep`` are injectable for fast tests.
    """

    def __init__(
        self,
        config: ThermalConfig | dict[str, Any] | None = None,
        *,
        sensor: SensorFn | None = None,
        sensor_names: list[str] | None = None,
        is_cuda: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config if isinstance(config, ThermalConfig) else ThermalConfig.from_dict(config)
        self._clock = clock
        self._sleep = sleep
        self._hot = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest = ThermalSample()
        self._reason = ""
        self.sensor_names: list[str] = list(sensor_names or [])
        self.failures = 0
        self.samples = 0
        self.pause_count = 0
        self.pause_seconds = 0.0
        self.max_gpu_temp_epoch: float | None = None
        self.max_gpu_temp_run: float | None = None

        self._sensor: SensorFn | None = None
        if not self.config.enabled:
            logger.info("thermal guard disabled by config")
        elif sensor is not None:
            self._sensor = sensor
            if not self.sensor_names:
                self.sensor_names = ["custom"]
        elif not is_cuda:
            logger.info("thermal guard inactive: device is CPU")
        else:
            self._sensor, self.sensor_names = build_sensor(want_gpu=True, want_cpu=True)
            if self._sensor is None:
                logger.info("thermal guard inactive: no temperature sensor available")
        if self._sensor is not None:
            c = self.config
            logger.info(
                "thermal guard active: sensors %s | every %g s | gpu pause>=%.0f resume<=%.0f C | "
                "cpu pause>=%.0f resume<=%.0f C | hw slowdown pause=%s | warn every %.0f min",
                "+".join(self.sensor_names), c.interval_s, c.gpu_pause_c, c.gpu_resume_c,
                c.cpu_pause_c, c.cpu_resume_c, c.pause_on_hw_slowdown, c.max_pause_min,
            )

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        return self._sensor is not None

    @property
    def hot(self) -> bool:
        """Cheap flag read for the training loop."""

        return self._hot.is_set()

    @property
    def latest(self) -> ThermalSample:
        return self._latest

    @property
    def reason(self) -> str:
        return self._reason

    def status(self) -> str:
        """Short human-readable sensor summary (``gpu 65 C`` / ``gpu 65 C cpu 70 C``)."""

        s = self._latest
        parts = []
        if s.gpu_temp_c is not None:
            parts.append(f"gpu {s.gpu_temp_c:.0f} C")
        if s.cpu_temp_c is not None:
            parts.append(f"cpu {s.cpu_temp_c:.0f} C")
        if s.hw_slowdown:
            parts.append("hw-slowdown")
        return " ".join(parts) if parts else "no sensor"

    def scalars(self) -> dict[str, float]:
        """TensorBoard ``sys/*`` values."""

        out: dict[str, float] = {"thermal_pause_s": float(self.pause_seconds)}
        s = self._latest
        if s.gpu_temp_c is not None:
            out["gpu_temp_c"] = float(s.gpu_temp_c)
        if s.cpu_temp_c is not None:
            out["cpu_temp_c"] = float(s.cpu_temp_c)
        return out

    def reset_epoch_stats(self) -> None:
        self.max_gpu_temp_epoch = None

    # ------------------------------------------------------------------ sampling
    def update(self, raw: dict[str, Any]) -> bool:
        """Apply one reading to the hysteresis state; returns the new *hot* flag."""

        sample = ThermalSample.from_dict(raw, now=self._clock())
        c = self.config
        if sample.gpu_temp_c is not None:
            self.max_gpu_temp_epoch = sample.gpu_temp_c if self.max_gpu_temp_epoch is None else max(self.max_gpu_temp_epoch, sample.gpu_temp_c)
            self.max_gpu_temp_run = sample.gpu_temp_c if self.max_gpu_temp_run is None else max(self.max_gpu_temp_run, sample.gpu_temp_c)

        reasons: list[str] = []
        if sample.gpu_temp_c is not None and sample.gpu_temp_c >= c.gpu_pause_c:
            reasons.append(f"gpu {sample.gpu_temp_c:.0f} C >= {c.gpu_pause_c:.0f} C")
        if sample.cpu_temp_c is not None and sample.cpu_temp_c >= c.cpu_pause_c:
            reasons.append(f"cpu {sample.cpu_temp_c:.0f} C >= {c.cpu_pause_c:.0f} C")
        if c.pause_on_hw_slowdown and sample.hw_slowdown:
            reasons.append("hw thermal slowdown active")

        if reasons:
            hot = True
        elif self._hot.is_set():
            # Hysteresis: stay hot until every reporting sensor is back at/below its resume level.
            cool = True
            if sample.gpu_temp_c is not None and sample.gpu_temp_c > c.gpu_resume_c:
                cool = False
            if sample.cpu_temp_c is not None and sample.cpu_temp_c > c.cpu_resume_c:
                cool = False
            if c.pause_on_hw_slowdown and sample.hw_slowdown:
                cool = False
            hot = not cool
        else:
            hot = False

        with self._lock:
            self._latest = sample
            if hot and reasons:
                self._reason = ", ".join(reasons)
            if hot:
                self._hot.set()
            else:
                self._hot.clear()
        return hot

    def sample_once(self) -> ThermalSample | None:
        """Read the sensor and update the flag; sensor failures keep the previous state."""

        if self._sensor is None:
            return None
        try:
            raw = self._sensor()
        except Exception as exc:  # noqa: BLE001 - a flaky sensor must not kill the run
            self.failures += 1
            if self.failures in (1, 10, 100) or self.failures % 1000 == 0:
                logger.warning("thermal sensor read failed (%d so far): %s", self.failures, exc)
            return None
        self.samples += 1
        self.update(raw or {})
        return self._latest

    # ------------------------------------------------------------------ thread
    def start(self) -> None:
        if self._sensor is None or self._thread is not None:
            return
        self._stop.clear()
        self.sample_once()  # the first progress line already carries a temperature
        self._thread = threading.Thread(target=self._run, name="xtx4-thermal", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.config.interval_s):
            self.sample_once()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, 2 * self.config.interval_s + _SUBPROCESS_TIMEOUT_S))

    # ------------------------------------------------------------------ wait
    def wait_until_cool(self, *, before_pause: Callable[[], None] | None = None) -> float:
        """Block while hot; returns the seconds spent paused (0 when nothing happened)."""

        if not self._hot.is_set():
            return 0.0
        if before_pause is not None:
            before_pause()
        t0 = self._clock()
        logger.info("thermal pause: %s", self._reason or self.status())
        warn_every = self.config.max_pause_min * 60.0
        last_warn = t0
        while self._hot.is_set() and not self._stop.is_set():
            self._sleep(self.config.interval_s)
            now = self._clock()
            if warn_every > 0 and now - last_warn >= warn_every:
                last_warn = now
                logger.warning(
                    "thermal pause exceeds %.0f min (%.0f s so far, %s); still waiting",
                    self.config.max_pause_min, now - t0, self.status(),
                )
        paused = self._clock() - t0
        self.pause_count += 1
        self.pause_seconds += paused
        logger.info("thermal resume after %.0f s (%s)", paused, self.status())
        return paused

    def __enter__(self) -> "ThermalGuard":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


__all__ = [
    "ThermalConfig",
    "ThermalGuard",
    "ThermalSample",
    "build_sensor",
    "read_nvidia_smi",
    "read_psutil_cpu",
    "read_wmi_cpu",
]
