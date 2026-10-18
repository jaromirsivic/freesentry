"""XTX4 test app (``live_test/``): settings and presets, the engine on a still image
(graph reloads, decode attributes, pause / step / redetect) and a GUI smoke test.

Uses the exported models in ``models/<variant>/deploy`` (skipped when they are
missing) and the first image of the XTX3 smoke dataset.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / "live_test"
SMOKE_DIR = REPO.parent / "xtx3" / ".smoke_dataset" / "test"

sys.path.insert(0, str(LIVE))
try:
    from livetest import settings as st  # noqa: E402
    from livetest.config import ConfigError, Preset, load_config  # noqa: E402
    from livetest.engine import LiveEngine  # noqa: E402
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def config():
    cfg = load_config(LIVE / "config.json")
    if not cfg.network("u_b").available_sizes():
        pytest.skip("models/u_b/deploy not exported")
    return cfg


@pytest.fixture(scope="module")
def image_path() -> str:
    paths = sorted(SMOKE_DIR.glob("*.jpg"))
    if not paths:
        pytest.skip("smoke dataset not available")
    return str(paths[0])


# ------------------------------------------------------------------ settings
def test_config_defaults_take_decode_values_from_the_manifest(config) -> None:
    settings = st.config_defaults(config)
    decode = config.network(settings.network_id).decode_defaults()
    assert settings.network_id == "u_b" and settings.size == 320 and settings.backend == "ncnn"
    assert settings.present_threshold == decode["present_threshold"]
    assert settings.oks_nms_scale_factor == decode["oks_nms_scale_factor"]
    assert settings.oks_nms_iou is (decode["oks_nms_iou_threshold"] is not None)


def test_from_dict_coerces_clamps_and_ignores_unknown_keys() -> None:
    base = st.Settings(network_id="u_b")
    out = st.from_dict({"conf_threshold": "1.7", "size": "384", "oks_nms": "true", "mode": "bogus",
                        "num_threads": 0, "unknown": 1, "fp16": 0}, base)
    assert out.conf_threshold == 1.0 and out.size == 384 and out.oks_nms is True and out.fp16 is False
    assert out.mode == "whole" and out.num_threads == 1


def test_organ_display_settings_from_config(config) -> None:
    from livetest.config import flatten_organs

    settings = st.config_defaults(config)
    assert settings.overlay_mode == "organs"  # decision O3: the live view starts with the organ circles
    assert settings.brain_conf_threshold == 0.25 and settings.chest_conf_threshold == 0.143
    assert settings.brain_enabled and settings.chest_enabled and settings.chest_min_radius == 1
    flat = flatten_organs({"brain": {"enabled": False, "sizeMultiplier": 2, "confidenceThreshold": 0.3,
                                     "minimumRadius": 5, "bogus": 1}, "heart": {"enabled": True}})
    assert flat == {"brain_enabled": False, "brain_size_multiplier": 2, "brain_conf_threshold": 0.3,
                    "brain_min_radius": 5}
    out = st.from_dict({"overlay_mode": "bogus", "chest_conf_threshold": 1.5, "brain_size_multiplier": 50}, settings)
    assert out.overlay_mode == "organs" and out.chest_conf_threshold == 1.0 and out.brain_size_multiplier == 5.0
    by_name = {p.name: p for p in config.presets}
    assert st.apply_preset(settings, by_name["Keypoints only"], config).overlay_mode == "keypoints"
    assert st.apply_preset(settings, by_name["Clean overlay"], config).overlay_mode == "keypoints"
    assert st.apply_preset(settings, by_name["Debug overlay (sigma, NMS extents, labels, network input)"],
                           config).overlay_mode == "both"


def _organ_pose():
    import numpy as np

    from xtx4.runtime import Pose

    kp = np.array([[300, 150], [290, 140], [310, 140], [275, 145], [325, 145],
                   [250, 230], [350, 230], [265, 380], [335, 380]], dtype=np.float32)
    kp = np.concatenate([kp, np.full((9, 1), 0.9, np.float32)], axis=1)
    organs = np.array([[300, 120, 40, 0.9], [300, 260, 70, 0.05]], dtype=np.float32)  # chest below threshold
    return Pose(score=0.9, keypoints=kp, visibility=np.full(9, 2, np.int64), sigma_px=np.ones(9, np.float32),
                organs=organs)


def test_draw_organs_freesentry_style_and_overlay_modes(config) -> None:
    import numpy as np

    sys.path.insert(0, str(LIVE))
    try:
        from livetest.draw import SIDE_COLORS, DrawOptions, draw_aicircle, draw_poses
    finally:
        sys.path.pop(0)
    skeleton = config.network("u_b").skeleton

    # draw_aicircle: large circle = white rim + colour ring, small = filled white dot, below threshold = thin
    frame = np.zeros((480, 640, 3), np.uint8)
    draw_aicircle(frame, cx=200, cy=200, radius=60, confidence=0.9, confidence_achieved=True,
                  size_multiplier=1.0, minimum_radius=1)
    assert (frame[200, 200 - 60] == 255).all() and not (frame[200, 200 - 56] == 255).all()  # rim + colour ring
    small = np.zeros((480, 640, 3), np.uint8)
    draw_aicircle(small, cx=100, cy=100, radius=5, confidence=0.9, confidence_achieved=True,
                  size_multiplier=1.0, minimum_radius=1)
    assert (small[100, 100] == 255).all()  # filled white dot
    thin = np.zeros((480, 640, 3), np.uint8)
    draw_aicircle(thin, cx=100, cy=100, radius=30, confidence=0.9, confidence_achieved=False,
                  size_multiplier=1.0, minimum_radius=1)
    assert not (thin == 255).all(axis=2).any() and thin[100, 130].any()  # colour ring only, no white

    def render(mode: str) -> np.ndarray:
        canvas = np.zeros((480, 640, 3), np.uint8)
        draw_poses(canvas, [_organ_pose()], skeleton, DrawOptions(overlay_mode=mode, show_scores=False))
        return canvas

    def has_colour(img: np.ndarray, bgr) -> bool:
        return bool((img == np.array(bgr, np.uint8)).all(axis=2).any())

    organs, keypoints, both = render("organs"), render("keypoints"), render("both")
    assert not has_colour(organs, SIDE_COLORS["l"]) and not has_colour(organs, SIDE_COLORS["r"])
    assert (organs == 255).all(axis=2).any()  # the brain's white rim
    assert has_colour(keypoints, SIDE_COLORS["l"]) and not (keypoints[120, 300 - 37] == 255).all()
    assert has_colour(both, SIDE_COLORS["l"]) and (both == 255).all(axis=2).any()
    hidden = np.zeros((480, 640, 3), np.uint8)
    draw_poses(hidden, [_organ_pose()], skeleton,
               DrawOptions(overlay_mode="organs", show_scores=False, brain_enabled=False, chest_enabled=False))
    assert not hidden.any()


def test_every_builtin_preset_applies_to_a_valid_combination(config) -> None:
    base = st.config_defaults(config)
    for preset in config.presets:
        settings = st.apply_preset(base, preset, config)
        network = config.network(settings.network_id)
        assert settings.size in network.available_sizes(settings.backend), preset.name
        for key, value in preset.settings.items():
            if key != "decode":
                assert getattr(settings, key) == value, (preset.name, key)


def test_manifest_decode_preset_resets_decode_but_keeps_the_model(config) -> None:
    tuned = st.config_defaults(config).updated(network_id="u_c", size=384, conf_threshold=0.1,
                                               visible_threshold=0.8, oks_nms=True)
    raw = st.apply_preset(tuned, st.find_preset("Raw network output", config.presets), config)
    assert (raw.network_id, raw.size) == ("u_c", 384)
    assert (raw.conf_threshold, raw.visible_threshold, raw.oks_nms) == (0.25, 0.5, False)


def test_unknown_preset_network_and_unavailable_backend_fall_back(config) -> None:
    base = st.config_defaults(config)
    out = st.apply_preset(base, Preset(name="x", settings={"network_id": "nope", "conf_threshold": 0.4}), config)
    assert out.network_id == base.network_id and out.conf_threshold == 0.4
    random = st.validate_for_config(base.updated(network_id="u_b-random", backend="onnxruntime"), config)
    assert random.backend == "ncnn"  # bench graphs have no ONNX
    with pytest.raises(ConfigError):
        st.find_preset("no such preset", config.presets)


def test_user_presets_and_last_settings_round_trip(config, tmp_path: Path) -> None:
    settings = st.config_defaults(config).updated(size=384, show_sigma=True, source_kind="image",
                                                  source_path="x.jpg")
    assert "source_path" not in settings.preset_dict()
    path = tmp_path / "user_presets.json"
    st.save_user_presets([Preset(name="mine", settings=settings.preset_dict(), builtin=False)], path)
    loaded = st.load_user_presets(path)
    assert [p.name for p in loaded] == ["mine"] and not loaded[0].builtin
    assert st.apply_preset(st.config_defaults(config), loaded[0], config).show_sigma is True
    last = tmp_path / "last.json"
    st.save_last(settings, last)
    assert st.load_last(config, last) == settings
    last.write_text("{broken", encoding="utf-8")
    assert st.load_last(config, last) == st.config_defaults(config)


# -------------------------------------------------------------------- engine
def _wait_result(engine: LiveEngine, after_seq: int = 0, timeout: float = 20.0):
    end = time.time() + timeout
    while time.time() < end:
        item = engine.latest()
        if item is not None and item.seq > after_seq:
            return item
        time.sleep(0.02)
    raise AssertionError("no pipeline result")


def test_engine_reloads_graph_sets_decode_attrs_and_pauses(config, image_path: str) -> None:
    settings = st.config_defaults(config).updated(source_kind="image", source_path=image_path, camera_fps=60,
                                                  conf_threshold=0.2, mirror=False)
    engine = LiveEngine(config)
    try:
        messages = engine.apply(settings)
        assert any(m.startswith("Loaded") for m in messages) and engine.source_error is None
        first = _wait_result(engine)
        assert first.result.poses, "smoke image has people"
        runner = engine.runner

        engine.apply(settings.updated(visible_threshold=0.9, mode="pyramid", conf_threshold=0.3))
        assert engine.runner is runner  # no reload for decode attributes
        assert (runner.visible_threshold, runner.mode) == (0.9, "pyramid")
        item = _wait_result(engine, first.seq + 2)
        assert all(p.score >= 0.3 for p in item.result.poses)

        engine.apply(engine.settings.updated(size=256, backend="onnxruntime", mode="whole"))
        assert engine.runner is not runner and engine.runner.backend == "onnxruntime" and engine.runner.size == 256
        assert engine.runner.visible_threshold == 0.9  # carried into the new graph
        item = _wait_result(engine, item.seq)

        engine.set_paused(True)
        time.sleep(0.3)
        frozen = engine.latest().seq
        time.sleep(0.3)
        assert engine.latest().seq <= frozen + 1  # at most the frame already in flight
        held = engine.latest().seq
        engine.step()
        assert _wait_result(engine, held, timeout=5).seq > held
        again = engine.redetect(item.frame)
        assert len(again.poses) == len(item.result.poses) or again.poses

        engine.apply(engine.settings.updated(source_path=str(Path(image_path).with_name("missing.jpg"))))
        assert engine.source_error and engine.pipeline is None
    finally:
        engine.close()


# ----------------------------------------------------------------------- GUI
def test_gui_smoke(config, image_path: str, tmp_path: Path, monkeypatch) -> None:
    tk = pytest.importorskip("tkinter")
    pytest.importorskip("PIL")
    sys.path.insert(0, str(LIVE))
    try:
        from livetest import gui
    finally:
        sys.path.pop(0)
    monkeypatch.setattr(gui, "list_cameras", lambda **_kw: [])
    monkeypatch.setattr(gui, "SCREENSHOT_DIR", tmp_path / "shots")
    monkeypatch.setattr(gui, "RECORDING_DIR", tmp_path / "rec")
    monkeypatch.setattr(gui, "save_last", lambda *_a, **_k: None)
    monkeypatch.setattr(gui, "save_user_presets", lambda presets: None)
    settings = st.config_defaults(config).updated(source_kind="image", source_path=image_path, mirror=False)
    try:
        app = gui.TestApp(config, settings)
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")

    def pump(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.root.update()
            time.sleep(0.01)

    try:
        pump(1.5)
        assert app._result is not None and app.display_rate._count + app.display_rate.value > 0  # frames shown
        assert not app._placeholder
        assert app.pose_tree.get_children(), "pose table filled"
        assert app.engine.settings.overlay_mode == "organs"
        assert all(p.organs is not None for p in app._result.poses)
        app.vars["overlay_mode"].set("both")
        app.vars["chest_size_multiplier"].set(1.5)
        pump(0.3)
        assert app._display_settings().overlay_mode == "both"
        app.selected_pose = 0
        app._update_pose_tables()
        names = [app.kpt_tree.item(i, "values")[0] for i in app.kpt_tree.get_children()]
        assert names[-2:] == ["brain", "chest"]

        # Model / detection / display changes go through the widgets' variables.
        app.vars["size"].set(384)
        app.vars["oks_nms"].set(True)
        app.vars["show_sigma"].set(True)
        app.vars["show_debug_view"].set(True)
        pump(1.0)
        assert app.engine.settings.size == 384 and app.engine.settings.oks_nms
        assert app.engine.runner.size == 384

        app.preset_box.current(next(i for i, p in enumerate(app.presets) if p.name.startswith("FP32 reference")))
        app._apply_preset()
        pump(1.0)
        assert app.engine.runner.backend == "onnxruntime"

        app.toggle_pause()
        app.vars["conf_threshold"].set(0.9)
        pump(0.6)
        assert app.engine.paused and all(p.score >= 0.9 for p in app._result.poses)  # frozen frame re-detected
        app.toggle_pause()

        app.screenshot()
        shots = list((tmp_path / "shots").glob("*.json"))
        assert len(shots) == 1 and "poses" in json.loads(shots[0].read_text(encoding="utf-8"))
        shot_poses = json.loads(shots[0].read_text(encoding="utf-8"))["poses"]
        assert not shot_poses or [o["name"] for o in shot_poses[0]["organs"]] == ["brain", "chest"]
        app.toggle_record()
        pump(0.5)
        app.toggle_record()
        assert list((tmp_path / "rec").glob("*.mp4"))
    finally:
        app.quit()
