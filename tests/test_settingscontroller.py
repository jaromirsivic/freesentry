import asyncio
import importlib
import json
import sys
import threading
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


def _write_settings(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def _read_settings(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_settings() -> dict:
    return {
        "general": {},
        "motors": [],
        "hotZone": {},
        "aiSetup": {},
        "manualControl": {},
        "cameras": {
            "scope_camera": {
                "index": 0,
                "static_reticle_x": 0.5,
                "static_reticle_y": 0.5,
            }
        },
        "primaryCamera": {},
    }


class SettingsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = TemporaryDirectory()
        self.settings_path = Path(self._temp_dir.name) / "settings.json"
        _write_settings(self.settings_path, _make_settings())
        self.settingscontroller = importlib.import_module("server.settingscontroller")

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def test_sync_snapshot_is_immutable_and_does_not_leak_cache(self):
        store = self.settingscontroller.SettingsStore(self.settings_path)

        snapshot = store.get_snapshot()
        mutable_copy = store.get_copy()
        mutable_copy["general"]["audit"] = "copy-only"

        with self.assertRaises(TypeError):
            snapshot["general"]["audit"] = "not-allowed"  # type: ignore[index]

        fresh_copy = store.get_copy()
        self.assertNotIn("audit", fresh_copy["general"])

    def test_save_failure_keeps_disk_and_cache_consistent(self):
        store = self.settingscontroller.SettingsStore(self.settings_path)
        original_settings = store.get_copy()

        updated_settings = store.get_copy()
        updated_settings["general"]["audit"] = "updated"

        with mock.patch("server.settingscontroller.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                store.save(updated_settings)

        self.assertEqual(_read_settings(self.settings_path), original_settings)
        self.assertEqual(store.get_copy(), original_settings)

    def test_concurrent_updates_do_not_lose_changes(self):
        store = self.settingscontroller.SettingsStore(self.settings_path)
        worker_count = 5
        updates_per_worker = 20
        start_barrier = threading.Barrier(worker_count)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                start_barrier.wait(timeout=5)
                for _ in range(updates_per_worker):
                    def increment_counter(settings: dict) -> None:
                        general = settings.setdefault("general", {})
                        general["counter"] = general.get("counter", 0) + 1

                    store.update(increment_counter)
            except Exception as exc:  # pragma: no cover - assertion below reports details
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(worker_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(errors, f"Unexpected worker errors: {errors}")
        persisted_settings = _read_settings(self.settings_path)
        self.assertEqual(
            persisted_settings["general"]["counter"],
            worker_count * updates_per_worker,
        )
        self.assertEqual(store.get_copy()["general"]["counter"], worker_count * updates_per_worker)

    def test_ai_setup_get_does_not_mutate_store_state(self):
        fake_ultralytics = types.ModuleType("ultralytics")

        class FakeYOLO:
            def __init__(self, *args, **kwargs):
                pass

        fake_ultralytics.YOLO = FakeYOLO

        sys.modules.pop("server.restapiaisetup", None)
        sys.modules.pop("server.yolomodels", None)

        with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            restapiaisetup = importlib.import_module("server.restapiaisetup")

        store = self.settingscontroller.SettingsStore(self.settings_path)

        with mock.patch.object(self.settingscontroller, "_SETTINGS_STORE", store):
            response = asyncio.run(restapiaisetup.get_ai_setup())

        self.assertIn("activationDateTime", response["aiSetup"])
        self.assertEqual(store.get_copy()["aiSetup"], {})
        self.assertEqual(_read_settings(self.settings_path)["aiSetup"], {})


if __name__ == "__main__":
    unittest.main()
