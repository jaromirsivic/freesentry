"""Thermal guard (xtx4/engine/thermal.py) with fake sensors: hysteresis, missing sensors, HW slowdown, pausing."""

from __future__ import annotations

import threading
import time

import pytest

from xtx4.engine.thermal import ThermalConfig, ThermalGuard, ThermalSample, read_nvidia_smi


class FakeSensor:
    """Scripted readings: the last entry repeats forever."""

    def __init__(self, readings: list[dict]) -> None:
        self.readings = list(readings)
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self) -> dict:
        with self.lock:
            idx = min(self.calls, len(self.readings) - 1)
            self.calls += 1
            return dict(self.readings[idx])


def _guard(sensor, **cfg) -> ThermalGuard:
    config = ThermalConfig.from_dict({"interval_s": 0.01, "max_pause_min": 60, **cfg})
    return ThermalGuard(config, sensor=sensor, is_cuda=False)


def test_config_defaults_and_validation():
    cfg = ThermalConfig.from_dict(None)
    assert cfg.enabled and cfg.interval_s == 10 and cfg.gpu_pause_c == 85 and cfg.gpu_resume_c == 75
    assert cfg.cpu_pause_c == 95 and cfg.cpu_resume_c == 85 and cfg.pause_on_hw_slowdown and cfg.max_pause_min == 60
    with pytest.raises(ValueError):
        ThermalConfig.from_dict({"gpu_pause_c": 70, "gpu_resume_c": 80})
    with pytest.raises(ValueError):
        ThermalConfig.from_dict({"interval_s": 0})


def test_gpu_hysteresis():
    guard = _guard(lambda: {})
    assert guard.active and not guard.hot
    assert guard.update({"gpu_temp_c": 84.9}) is False
    assert guard.update({"gpu_temp_c": 85.0}) is True, "pause threshold is inclusive"
    assert "gpu 85 C >= 85 C" in guard.reason
    assert guard.update({"gpu_temp_c": 80.0}) is True, "must stay hot between resume and pause thresholds"
    assert guard.update({"gpu_temp_c": 75.1}) is True
    assert guard.update({"gpu_temp_c": 75.0}) is False, "resume threshold is inclusive"
    assert guard.update({"gpu_temp_c": 80.0}) is False, "cool state keeps until the pause threshold"
    assert guard.max_gpu_temp_epoch == 85.0 and guard.max_gpu_temp_run == 85.0
    guard.reset_epoch_stats()
    assert guard.max_gpu_temp_epoch is None and guard.max_gpu_temp_run == 85.0


def test_cpu_and_gpu_both_must_cool_down():
    guard = _guard(lambda: {})
    assert guard.update({"gpu_temp_c": 60.0, "cpu_temp_c": 96.0}) is True
    assert "cpu 96 C >= 95 C" in guard.reason
    assert guard.update({"gpu_temp_c": 60.0, "cpu_temp_c": 86.0}) is True
    assert guard.update({"gpu_temp_c": 76.0, "cpu_temp_c": 85.0}) is True, "gpu above its resume level"
    assert guard.update({"gpu_temp_c": 75.0, "cpu_temp_c": 85.0}) is False
    assert guard.update({"cpu_temp_c": 95.0}) is True  # a missing gpu reading does not block a pause


def test_hw_slowdown_flag():
    guard = _guard(lambda: {}, pause_on_hw_slowdown=True)
    assert guard.update({"gpu_temp_c": 60.0, "hw_slowdown": True}) is True
    assert "hw thermal slowdown" in guard.reason
    assert guard.update({"gpu_temp_c": 60.0, "hw_slowdown": True}) is True
    assert guard.update({"gpu_temp_c": 60.0, "hw_slowdown": False}) is False
    off = _guard(lambda: {}, pause_on_hw_slowdown=False)
    assert off.update({"gpu_temp_c": 60.0, "hw_slowdown": True, "sw_slowdown": True}) is False
    assert off.latest.hw_slowdown and off.latest.sw_slowdown  # still recorded for the logs


def test_missing_sensors_never_pause():
    empty = _guard(lambda: {})
    assert empty.update({}) is False and empty.update({"gpu_temp_c": None, "cpu_temp_c": float("nan")}) is False
    assert empty.status() == "no sensor" and empty.scalars() == {"thermal_pause_s": 0.0}

    # No injected sensor on a CPU device: inactive, start()/wait are no-ops.
    inactive = ThermalGuard(ThermalConfig.from_dict({"interval_s": 0.01}), sensor=None, is_cuda=False)
    assert not inactive.active and not inactive.hot
    inactive.start()
    assert inactive._thread is None
    assert inactive.wait_until_cool() == 0.0
    inactive.stop()

    disabled = ThermalGuard(ThermalConfig.from_dict({"enabled": False}), sensor=lambda: {"gpu_temp_c": 99.0}, is_cuda=True)
    assert not disabled.active
    assert disabled.sample_once() is None and not disabled.hot


def test_sensor_failure_keeps_previous_state():
    calls = {"n": 0}

    def flaky() -> dict:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("nvidia-smi exploded")
        return {"gpu_temp_c": 90.0}

    guard = _guard(flaky)
    assert guard.sample_once() is not None and guard.hot
    assert guard.sample_once() is None and guard.hot and guard.failures == 1
    assert guard.sample_once() is not None and guard.samples == 2


def test_thread_pauses_then_resumes():
    sensor = FakeSensor([{"gpu_temp_c": 90.0}] * 4 + [{"gpu_temp_c": 70.0}])
    guard = _guard(sensor)
    synced = []
    with guard:
        assert guard.hot, "start() samples synchronously so the first step already sees the flag"
        t0 = time.perf_counter()
        paused = guard.wait_until_cool(before_pause=lambda: synced.append(True))
        wall = time.perf_counter() - t0
    assert synced == [True]
    assert paused > 0.0 and abs(paused - wall) < 0.5
    assert guard.pause_count == 1 and guard.pause_seconds == pytest.approx(paused)
    assert not guard.hot and guard.latest.gpu_temp_c == 70.0
    assert guard.scalars()["gpu_temp_c"] == 70.0 and guard.scalars()["thermal_pause_s"] == pytest.approx(paused)
    assert guard.wait_until_cool() == 0.0 and guard.pause_count == 1
    assert guard._thread is None  # stopped cleanly


def test_long_pause_warns_and_keeps_waiting(caplog):
    clock = {"t": 1000.0}
    sleeps: list[float] = []

    def fake_clock() -> float:
        return clock["t"]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["t"] += 60.0  # every sleep advances a full minute

    config = ThermalConfig.from_dict({"interval_s": 10, "max_pause_min": 1})
    guard = ThermalGuard(config, sensor=lambda: {"gpu_temp_c": 90.0}, is_cuda=False, clock=fake_clock, sleep=fake_sleep)
    guard.update({"gpu_temp_c": 90.0})
    releaser = threading.Thread(target=lambda: (time.sleep(0.2), guard.update({"gpu_temp_c": 70.0})))
    releaser.start()
    with caplog.at_level("WARNING", logger="xtx4.engine.thermal"):
        paused = guard.wait_until_cool()
    releaser.join()
    assert paused >= 60.0 and sleeps and all(s == 10 for s in sleeps)
    assert any("thermal pause exceeds 1 min" in rec.getMessage() for rec in caplog.records)


def test_sample_dataclass_from_dict():
    sample = ThermalSample.from_dict({"gpu_temp_c": "41", "hw_slowdown": 1}, now=5.0)
    assert sample.gpu_temp_c == 41.0 and sample.cpu_temp_c is None and sample.hw_slowdown and sample.time == 5.0


def test_nvidia_smi_parser_on_real_machine_if_present():
    try:
        reading = read_nvidia_smi()
    except RuntimeError as exc:
        pytest.skip(f"nvidia-smi unavailable: {exc}")
    assert 0.0 < reading["gpu_temp_c"] < 150.0
    assert isinstance(reading["hw_slowdown"], bool) and isinstance(reading["sw_slowdown"], bool)
