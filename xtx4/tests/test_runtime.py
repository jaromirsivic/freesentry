"""Torch-free runtime tests: NumPy decode vs torch, letterbox parity, ncnn runner vs
onnxruntime, OKS-NMS, pipeline, manifest, deployment export and the live test.

Artefacts are written under ``D:/xtx4_runs/_tests/`` (drive C: is nearly full) and
removed afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

import cv2
import numpy as np
import pytest
import torch

from xtx4.data import preprocess as data_preprocess
from xtx4.export.manifest import (
    DECODE_VERSION,
    ManifestError,
    build_manifest,
    load_manifest,
    write_manifest,
)
from xtx4.models.head_types import RawPredictions
from xtx4.models.head_types import build_anchors as torch_build_anchors
from xtx4.models.head_types import decode_detections
from xtx4.runtime import (
    InferencePipeline,
    LatestFrameQueue,
    NcnnPoseRunner,
    Pose,
    build_anchors,
    decode_raw,
    letterbox,
    oks_nms,
    prepare_views,
    run_headless,
    to_blob,
    to_frame_poses,
)
from xtx4.runtime.pipeline import FramePacket

REPO = Path(__file__).resolve().parents[1]
GRAPHS = REPO / "bench" / "graphs"
SMOKE_DIR = REPO.parent / "xtx3" / ".smoke_dataset" / "test"
K = 9
S = 320
STRIDES = (8, 16, 32)
N320 = sum((S // s) ** 2 for s in STRIDES)  # 2100


def _smoke_image() -> np.ndarray:
    paths = sorted(SMOKE_DIR.glob("*.jpg"))
    if not paths:
        pytest.skip("smoke dataset not available")
    image = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    assert image is not None
    return image


@pytest.fixture()
def artefact_dir(tmp_path: Path):
    root = Path("D:/xtx4_runs/_tests")
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"runtime-{uuid.uuid4().hex[:8]}"
        path.mkdir()
    except OSError:
        path = tmp_path / "artefacts"
        path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ------------------------------------------------------------------ decode
@pytest.mark.parametrize("size", [256, 320, 384])
@pytest.mark.parametrize("strides", [(8, 16, 32), (8, 32), (16, 32)])
def test_build_anchors_matches_torch(size: int, strides: tuple[int, ...]) -> None:
    anchors, stride_arr = build_anchors(size, strides)
    sizes = [(size // s, size // s) for s in strides]
    t_anchors, t_strides = torch_build_anchors(
        feature_sizes=sizes, strides=list(strides), device=torch.device("cpu"), dtype=torch.float32
    )
    assert anchors.dtype == np.float32 and anchors.shape == (sum(h * w for h, w in sizes), 2)
    np.testing.assert_array_equal(anchors, t_anchors.numpy())
    np.testing.assert_array_equal(stride_arr, t_strides.numpy())
    assert build_anchors(size, list(strides))[0] is anchors  # cached
    assert not anchors.flags.writeable


def _random_raw(seed: int, thresholds: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Raw (person (1,N,1), pose (1,N,45)) whose scores stay away from the thresholds."""

    rng = np.random.default_rng(seed)
    person = rng.normal(-1.0, 2.0, size=(1, N320, 1)).astype(np.float32)
    logits_at_thr = [np.log(t / (1.0 - t)) for t in thresholds]
    for value in logits_at_thr:  # keep rounding of sigmoid away from the decision boundary
        close = np.abs(person - value) < 1e-3
        person[close] += np.float32(0.01)
    pose = rng.normal(0.0, 1.5, size=(1, N320, K, 5)).astype(np.float32)
    pose[..., 2] = rng.normal(0.0, 8.0, size=(1, N320, K))  # exercise both softplus branches (> 20)
    pose[0, :50, :, 3] = 0.0  # present exactly 0.5 -> visibility boundary
    return person, pose.reshape(1, N320, K * 5)


def _torch_decode(person: np.ndarray, pose: np.ndarray, **kw):
    anchors, strides = torch_build_anchors(
        feature_sizes=[(S // s, S // s) for s in STRIDES], strides=list(STRIDES),
        device=torch.device("cpu"), dtype=torch.float32,
    )
    raw = RawPredictions(
        person=torch.from_numpy(person), kpt=torch.from_numpy(pose).reshape(1, N320, K, 5),
        anchors=anchors, strides=strides,
    )
    dets, vis, sig = decode_detections(raw, **kw)
    return dets[0].numpy(), vis[0].numpy(), sig[0].numpy()


THRESHOLDS = (0.05, 0.25, 0.5, 0.9)


@pytest.mark.parametrize("conf_threshold", THRESHOLDS)
@pytest.mark.parametrize("max_detections", [100, 7])
@pytest.mark.parametrize("sigma_conf_coeff", [0.0, 0.3])
def test_decode_matches_torch(conf_threshold: float, max_detections: int, sigma_conf_coeff: float) -> None:
    person, pose = _random_raw(3, THRESHOLDS)
    dets, vis_t, sig_t = _torch_decode(
        person, pose, conf_threshold=conf_threshold, max_detections=max_detections, sigma_conf_coeff=sigma_conf_coeff
    )
    scores, kpts, vis, sig = decode_raw(
        person, pose, network_size=S, strides=STRIDES, num_keypoints=K, conf_threshold=conf_threshold,
        max_detections=max_detections, sigma_conf_coeff=sigma_conf_coeff,
    )
    expected_count = min(max_detections, int((1 / (1 + np.exp(-person.astype(np.float64))) >= conf_threshold).sum()))
    assert scores.shape == (expected_count,) and dets.shape[0] == expected_count
    assert kpts.shape == (expected_count, K, 3) and vis.shape == (expected_count, K) and sig.shape == (expected_count, K)
    assert kpts.dtype == np.float32 and vis.dtype == np.int64 and sig.dtype == np.float32
    assert np.all(np.diff(scores) <= 0)
    np.testing.assert_allclose(scores, dets[:, 0], rtol=0, atol=1e-6)
    t_kpts = dets[:, 1:].reshape(-1, K, 3)
    np.testing.assert_array_equal(kpts[..., :2], t_kpts[..., :2])  # xy: identical float32 ops
    np.testing.assert_allclose(kpts[..., 2], t_kpts[..., 2], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(vis, vis_t)
    np.testing.assert_allclose(sig, sig_t, rtol=1e-5, atol=1e-5)


def test_decode_empty_and_shape_errors() -> None:
    person = np.full((N320, 1), -10.0, np.float32)
    pose = np.zeros((N320, K * 5), np.float32)
    scores, kpts, vis, sig = decode_raw(person, pose, network_size=S, strides=STRIDES, num_keypoints=K,
                                        conf_threshold=0.25, max_detections=10)
    assert scores.shape == (0,) and kpts.shape == (0, K, 3) and vis.shape == (0, K) and sig.shape == (0, K)
    with pytest.raises(ValueError):
        decode_raw(person[:-1], pose, network_size=S, strides=STRIDES, num_keypoints=K, conf_threshold=0.2,
                   max_detections=5)
    with pytest.raises(ValueError):
        decode_raw(person, pose[:, :-5], network_size=S, strides=STRIDES, num_keypoints=K, conf_threshold=0.2,
                   max_detections=5)


def test_decode_tie_order_by_anchor_index() -> None:
    """Equal scores keep ascending anchor order (stable). torch's argsort is not stable
    for ties, so against torch only the decoded *set* is compared."""

    person = np.full((1, N320, 1), -6.0, np.float32)
    tie_anchors = np.array([1500, 3, 2099, 700, 42])
    person[0, tie_anchors, 0] = 1.0
    person[0, 999, 0] = 2.0
    rng = np.random.default_rng(0)
    pose = rng.normal(0, 1, (1, N320, K * 5)).astype(np.float32)
    anchors, _ = build_anchors(S, STRIDES)
    scores, kpts, _, _ = decode_raw(person, pose, network_size=S, strides=STRIDES, num_keypoints=K,
                                    conf_threshold=0.5, max_detections=4)
    assert scores.shape == (4,)
    expected_order = [999, *sorted(tie_anchors.tolist())[:3]]
    p4 = pose.reshape(N320, K, 5)
    for row, anchor in enumerate(expected_order):
        stride = 8.0 if anchor < 1600 else 16.0 if anchor < 2000 else 32.0
        np.testing.assert_array_equal(kpts[row, :, 0], anchors[anchor, 0] + p4[anchor, :, 0] * np.float32(stride))
    # Same multiset as torch when all ties fit into max_detections.
    scores_all, kpts_all, _, _ = decode_raw(person, pose, network_size=S, strides=STRIDES, num_keypoints=K,
                                            conf_threshold=0.5, max_detections=100)
    dets, _, _ = _torch_decode(person, pose, conf_threshold=0.5, max_detections=100)
    ours = np.concatenate([scores_all[:, None], kpts_all.reshape(len(scores_all), -1)], axis=1)
    np.testing.assert_allclose(np.sort(ours, axis=0), np.sort(dets, axis=0), rtol=1e-6, atol=1e-6)


def test_to_frame_poses_applies_transform() -> None:
    scores = np.array([0.9], np.float32)
    kpts = np.zeros((1, K, 3), np.float32)
    kpts[0, :, 0] = np.arange(K) * 10.0
    kpts[0, :, 1] = 5.0
    kpts[0, :, 2] = 0.8
    vis = np.full((1, K), 2, np.int64)
    sig = np.full((1, K), 4.0, np.float32)
    transform = np.array([[2.0, 0.0, -8.0], [0.0, 2.0, 100.0]])
    (pose,) = to_frame_poses(scores, kpts, vis, sig, transform=transform, view=0, anchor_indices=np.array([17]))
    np.testing.assert_allclose(pose.keypoints[:, 0], np.arange(K) * 20.0 - 8.0)
    np.testing.assert_allclose(pose.keypoints[:, 1], 110.0)
    np.testing.assert_allclose(pose.keypoints[:, 2], 0.8, rtol=1e-6)
    np.testing.assert_allclose(pose.sigma_px, 8.0)
    assert pose.anchor_index == 17 and pose.score == pytest.approx(0.9)


# ------------------------------------------------------------------ OKS-NMS
def _pose(score: float, offset_x: float, *, jitter: float = 0.0) -> Pose:
    base = np.array(
        [[100, 50], [95, 45], [105, 45], [88, 48], [112, 48], [80, 90], [120, 90], [85, 170], [115, 170]],
        dtype=np.float32,
    )
    kp = np.concatenate([base + [offset_x + jitter, jitter], np.full((K, 1), 0.9, np.float32)], axis=1)
    return Pose(score=score, keypoints=kp.astype(np.float32), visibility=np.full(K, 2, np.int64),
                sigma_px=np.ones(K, np.float32))


def test_oks_nms_removes_injected_duplicate() -> None:
    a = _pose(0.9, 0.0)
    dup = _pose(0.6, 0.0, jitter=1.5)  # same person, slightly shifted, lower score
    other = _pose(0.8, 300.0)
    kept = oks_nms([dup, a, other], threshold=0.5)
    assert [p.score for p in kept] == [pytest.approx(0.9), pytest.approx(0.8)]
    # A pose far away is never suppressed; a strict threshold keeps everything.
    assert len(oks_nms([a, dup, other], threshold=1.01)) == 3
    assert oks_nms([], threshold=0.5) == []


# ---------------------------------------------------------------- letterbox
@pytest.mark.parametrize("seed", range(6))
def test_letterbox_matches_training_preprocess(seed: int) -> None:
    rng = np.random.default_rng(seed)
    height, width = (int(v) for v in rng.integers(50, 1400, size=2))
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    for size in (256, 320, 384):
        ours = prepare_views(image, network_size=size, mode="whole")
        ref = data_preprocess.preprocess_image(image, mode="whole", network_size=size)
        np.testing.assert_array_equal(ours.views[0], ref.views[0])
        np.testing.assert_array_equal(ours.transforms[0], ref.transforms[0])
        view, transform = letterbox(image, size)
        np.testing.assert_array_equal(view, ref.views[0])
        assert to_blob(view).shape == (1, 3, size, size)
    pyr = prepare_views(image, network_size=256, mode="pyramid")
    ref = data_preprocess.preprocess_image(image, mode="pyramid", network_size=256)
    for key in (1, 2, 3):
        np.testing.assert_array_equal(pyr.views[key], ref.views[key])
        np.testing.assert_array_equal(pyr.transforms[key], ref.transforms[key])


# ------------------------------------------------------------- ncnn runner
@pytest.fixture(scope="module")
def bench_runner():
    param, bin_path = GRAPHS / "u_b_320.ncnn.param", GRAPHS / "u_b_320.ncnn.bin"
    if not param.is_file():
        pytest.skip("bench graphs not exported")
    runner = NcnnPoseRunner(param, bin_path, size=320, num_threads=4, fp16=True)
    yield runner
    runner.close()


def test_ncnn_runner_detect_on_bench_graph(bench_runner: NcnnPoseRunner) -> None:
    assert bench_runner.strides == (8, 16, 32) and bench_runner.num_anchors == N320
    assert bench_runner.output_blobs == ("out0", "out1")
    image = _smoke_image()
    random_frame = np.random.default_rng(1).integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    for frame in (image, random_frame):
        view, transform = letterbox(frame, 320)
        person, pose = bench_runner.forward(view)
        assert person.shape == (N320, 1) and pose.shape == (N320, K * 5)
        res = bench_runner.detect(frame, conf_threshold=0.0, max_detections=30)
        assert res.raw_candidates == N320 and len(res.poses) == 30
        assert res.timings.preprocess_ms > 0 and res.timings.inference_ms > 0 and res.timings.postprocess_ms > 0
        assert all(p.keypoints.shape == (K, 3) and p.visibility.shape == (K,) for p in res.poses)
        # Frame-space poses == network decode mapped through T.
        scores, kpts, vis, sig = decode_raw(person, pose, network_size=320, strides=(8, 16, 32), num_keypoints=K,
                                            conf_threshold=0.0, max_detections=30)
        ref = to_frame_poses(scores, kpts, vis, sig, transform=transform)
        np.testing.assert_allclose([p.score for p in res.poses], [p.score for p in ref], atol=1e-6)
        np.testing.assert_allclose(np.stack([p.keypoints for p in res.poses]),
                                   np.stack([p.keypoints for p in ref]), atol=1e-3)
        high = bench_runner.detect(frame, conf_threshold=0.99, max_detections=30)
        assert len(high.poses) == high.raw_candidates == int((1 / (1 + np.exp(-person)) >= 0.99).sum())


def test_ncnn_runner_matches_onnxruntime(bench_runner: NcnnPoseRunner, record_property) -> None:
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(GRAPHS / "u_b_320.onnx"), providers=["CPUExecutionProvider"])
    image = _smoke_image()
    frames = {"smoke": image, "random": np.random.default_rng(2).integers(0, 256, (480, 640, 3), dtype=np.uint8)}
    worst = {}
    for name, frame in frames.items():
        view, _ = letterbox(frame, 320)
        person_o, pose_o = session.run(["person", "pose"], {"in0": to_blob(view)})
        person_n, pose_n = bench_runner.forward(view)
        d_person = float(np.abs(person_n.reshape(-1) - person_o.reshape(-1)).max())
        d_pose = float(np.abs(pose_n.reshape(-1) - pose_o.reshape(-1)).max())
        worst[name] = (d_person, d_pose)
        # fp16 storage/arithmetic tolerance on raw outputs (random weights, outputs O(1)).
        assert d_person < 2e-2 and d_pose < 2e-2, (name, d_person, d_pose)
    record_property("ncnn_vs_ort_max_abs", json.dumps(worst))
    print(f"ncnn(fp16) vs onnxruntime max |diff| (person, pose): {worst}")


def test_runner_rejects_bad_blobs_and_infers_strides() -> None:
    param, bin_path = GRAPHS / "u_b_P4P5_256.ncnn.param", GRAPHS / "u_b_P4P5_256.ncnn.bin"
    if not param.is_file():
        pytest.skip("bench graphs not exported")
    with NcnnPoseRunner(param, bin_path, size=256, strides=(8, 16, 32), output_blobs=("x", "y")) as runner:
        assert runner.strides == (16, 32)  # wrong strides corrected from N, blobs from the graph
        assert runner.output_blobs == ("out0", "out1")
    from xtx4.runtime import RunnerError

    with pytest.raises(RunnerError):
        NcnnPoseRunner(param, bin_path, size=256, num_keypoints=17)  # pose size != N * 17 * 5


# ----------------------------------------------------------------- pipeline
def test_latest_frame_queue_drops_old_frames() -> None:
    queue = LatestFrameQueue()
    frame = np.zeros((2, 2, 3), np.uint8)
    assert not queue.put(FramePacket(1, frame, time.perf_counter()))
    assert queue.put(FramePacket(2, frame, time.perf_counter()))
    packet = queue.get(timeout=0.1)
    assert packet is not None and packet.seq == 2 and queue.dropped == 1
    assert queue.get(timeout=0.01) is None


def test_pipeline_counts_drops_and_frame_age() -> None:
    from xtx4.runtime.ncnn_runner import DetectionResult, Timings

    frame = np.zeros((8, 8, 3), np.uint8)
    reads = {"n": 0}
    lock = threading.Lock()

    def read():
        time.sleep(0.002)
        with lock:
            reads["n"] += 1
        return frame

    def detect(_frame):
        time.sleep(0.01)  # slower than the source -> drops
        return DetectionResult(poses=[], timings=Timings(0.1, 9.0, 0.2))

    pipe = InferencePipeline(read, detect, max_frames=15).start()
    assert pipe.wait(10.0)
    pipe.stop()
    summary = pipe.stats.summary()
    assert pipe.error is None
    assert summary["processed"] == 15 and summary["dropped"] > 0
    assert summary["captured"] >= summary["processed"] + summary["dropped"] - 1
    assert summary["frame_age_ms"]["p50"] >= 9.0 and summary["inference_ms"]["p95"] == pytest.approx(9.0)
    assert "frame_age" in pipe.stats.report()
    stats = run_headless(read, detect, frames=5, warmup=2, timeout_s=10.0)
    assert stats.processed == 5


# ----------------------------------------------------------------- manifest
def _manifest() -> dict:
    return build_manifest(
        model_id="test-u_a", variant_spec={"name": "u_a"}, strides=[8, 16, 32], levels="P3P4P5", sizes=[256, 320],
        checkpoint_sha256="ab" * 32, config={"decode": {"conf_threshold": 0.3}},
        onnx_files={256: {"path": "xtx4_256.onnx"}, 320: {"path": "xtx4_320.onnx"}},
        ncnn_files={256: {"param": "xtx4_256.ncnn.param", "bin": "xtx4_256.ncnn.bin"}},
        decode={"conf_threshold": 0.3, "sigma_conf_coeff": 0.1}, tools={"pnnx": "x"},
    )


def test_manifest_round_trip(artefact_dir: Path) -> None:
    manifest = _manifest()
    for key in ("schema_version", "model_id", "checkpoint_sha256", "config_sha256", "variant_spec", "keypoints",
                "input", "strides", "levels", "anchor_order", "outputs", "decode_version", "decode", "precision",
                "tools", "created"):
        assert key in manifest
    assert manifest["decode_version"] == DECODE_VERSION == 1
    assert manifest["keypoints"]["names"][0] == "nose" and len(manifest["keypoints"]["skeleton_edges"]) == 8
    assert manifest["outputs"]["ncnn"]["files"]["256"]["shapes"] == {"person": [1344, 1], "pose": [1344, 45]}
    assert manifest["outputs"]["onnx"]["files"]["320"]["shapes"]["pose"] == [1, 2100, 45]
    assert manifest["decode"]["conf_threshold"] == 0.3 and manifest["decode"]["sigma_conf_coeff"] == 0.1
    path = write_manifest(artefact_dir / "manifest.json", manifest)
    assert load_manifest(path) == json.loads(json.dumps(manifest))
    assert load_manifest(artefact_dir) == load_manifest(path)
    bad = dict(manifest, decode_version=2)
    (artefact_dir / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(artefact_dir / "bad.json")
    missing = {k: v for k, v in manifest.items() if k != "strides"}
    (artefact_dir / "missing.json").write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(artefact_dir / "missing.json")


# ------------------------------------------------------------------- deploy
def _trained_like_u_a():
    """Random u_a with randomised BN statistics and a non-trivial pose head, so the
    raw outputs vary across anchors (a freshly initialised head is almost constant)."""

    from xtx4.models import build_model

    torch.manual_seed(0)
    model = build_model({"model": {"variant": "u_a"}})
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.copy_(torch.rand(m.num_features, generator=gen) * 0.2 - 0.1)
                m.running_var.copy_(torch.rand(m.num_features, generator=gen) + 0.5)
        for head in model.head.one2one.modules():
            if isinstance(head, torch.nn.Conv2d) and head.kernel_size == (1, 1) and head.out_channels in (1, K * 5):
                head.weight.normal_(0.0, 0.05, generator=gen)
    return model.eval()


def test_deploy_export_model_writes_graphs_manifest_and_parity(artefact_dir: Path) -> None:
    from xtx4.export.deploy import TOLERANCES, export_model
    from xtx4.export.onnx_ncnn import find_pnnx

    if find_pnnx() is None:
        pytest.skip("pnnx not installed")
    model = _trained_like_u_a()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = export_model(model, {"run": {"name": "t"}, "decode": {"conf_threshold": 0.25}}, artefact_dir,
                          sizes=[256], fp16=True)
    assert not model.fused and all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    for name in ("xtx4_256.onnx", "xtx4_256.ncnn.param", "xtx4_256.ncnn.bin", "manifest.json", "parity.json"):
        assert (artefact_dir / name).is_file(), name
    assert not list(artefact_dir.glob("*pnnx*"))  # intermediates removed
    manifest = load_manifest(artefact_dir / "manifest.json")
    assert manifest["strides"] == [8, 16, 32] and manifest["input"]["sizes"] == [256]
    assert manifest["outputs"]["ncnn"]["blobs"] == {"person": "out0", "pose": "out1"}
    parity = json.loads((artefact_dir / "parity.json").read_text(encoding="utf-8"))
    summary = parity["sizes"]["256"]["summary"]
    assert parity["pass"] and summary["pass"], summary
    for key, tol in TOLERANCES.items():
        assert summary[key] <= tol, (key, summary[key])
    assert len(parity["sizes"]["256"]["inputs"]) == 3
    print("deploy parity @256:", json.dumps(summary))
    with NcnnPoseRunner.from_manifest(artefact_dir / "manifest.json", 256) as runner:
        assert runner.strides == (8, 16, 32)
        res = runner.detect(_smoke_image(), conf_threshold=0.0, max_detections=5)
        assert len(res.poses) == 5
        runner.mode = "pyramid"  # view mode is a plain attribute: no reload needed
        pyramid = runner.detect(_smoke_image(), conf_threshold=0.0, max_detections=5)
        assert sorted(pyramid.prepared.views) == [1, 2, 3] and pyramid.raw_candidates == 3 * runner.num_anchors

    # The FP32 ONNX reference runs the same decode path as ncnn.
    pytest.importorskip("onnxruntime")
    from xtx4.runtime import OnnxPoseRunner

    with OnnxPoseRunner.from_manifest(artefact_dir / "manifest.json", 256, num_threads=2) as onnx_runner:
        assert onnx_runner.strides == (8, 16, 32) and onnx_runner.num_anchors == res.raw_candidates
        ref = onnx_runner.detect(_smoke_image(), conf_threshold=0.0, max_detections=5)
    assert len(ref.poses) == 5
    by_anchor = {p.anchor_index: p for p in ref.poses}
    common = [p for p in res.poses if p.anchor_index in by_anchor]
    assert len(common) >= 3
    for pose in common:
        other = by_anchor[pose.anchor_index]
        assert abs(pose.score - other.score) < 0.01
        assert np.abs(pose.keypoints[:, :2] - other.keypoints[:, :2]).max() < 2.0


# -------------------------------------------------------------- torch-free
def test_runtime_import_does_not_import_torch() -> None:
    code = (
        "import sys; import xtx4.runtime, xtx4.runtime.ncnn_runner, xtx4.runtime.pipeline, xtx4.export.manifest; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m.startswith('torch'))[:5]; print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and "ok" in proc.stdout, proc.stderr


def test_live_test_headless_runs_without_torch() -> None:
    if not (GRAPHS / "u_b_256.ncnn.param").is_file():
        pytest.skip("bench graphs not exported")
    image = sorted(SMOKE_DIR.glob("*.jpg"))
    if not image:
        pytest.skip("smoke dataset not available")
    proc = subprocess.run(
        [sys.executable, "main.py", "--headless", "--frames", "15", "--warmup", "2", "--image", str(image[0]),
         "--network", "u_b-random", "--size", "256", "--source-fps", "100", "--conf", "0.0"],
        cwd=str(REPO / "live_test"), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "torch imported: False" in proc.stdout
    assert "processed 15" in proc.stdout and "inference: p50" in proc.stdout


def test_oks_nms_matches_python_api() -> None:
    """The runtime OKS-NMS uses the same OKS scale proxy as ``xtx4.api.oks_nms``."""

    api = pytest.importorskip("xtx4.api")
    rng = np.random.default_rng(7)
    base = _pose(1.0, 0.0).keypoints
    poses = []
    for i in range(30):
        kp = base.copy()
        kp[:, :2] += rng.normal(0, 1, 2) * 40 + rng.normal(0, 3, (K, 2))
        kp[:, 2] = rng.uniform(0.2, 1.0, K)
        poses.append(Pose(score=float(rng.uniform()), keypoints=kp.astype(np.float32),
                          visibility=np.full(K, 2, np.int64), sigma_px=np.ones(K, np.float32), anchor_index=i))
    for thr in (0.3, 0.5, 0.8):
        ours = [p.anchor_index for p in oks_nms(poses, threshold=thr)]
        keep = api.oks_nms(np.stack([p.keypoints for p in poses]), np.array([p.score for p in poses]), threshold=thr)
        assert ours == keep.tolist(), thr
        assert 0 < len(ours) < len(poses) or thr == 0.8


# ------------------------------------------- visibility thresholds / NMS options
@pytest.mark.parametrize("present_threshold,visible_threshold", [(0.6, 0.8), (0.3, 0.2), (0.5, 0.5)])
def test_visibility_thresholds_match_torch(present_threshold: float, visible_threshold: float) -> None:
    from xtx4.models.head_types import decode_keypoints

    person, pose = _random_raw(11, (0.05,))
    kw = dict(conf_threshold=0.05, max_detections=200)
    th = dict(present_threshold=present_threshold, visible_threshold=visible_threshold)
    _, vis_t, _ = _torch_decode(person, pose, **kw, **th)
    _, kpts, vis, _ = decode_raw(person, pose, network_size=S, strides=STRIDES, num_keypoints=K, **kw, **th)
    np.testing.assert_array_equal(vis, vis_t)
    _, kpts_d, vis_d, _ = decode_raw(person, pose, network_size=S, strides=STRIDES, num_keypoints=K, **kw)
    np.testing.assert_array_equal(kpts, kpts_d)  # conf / xy do not depend on the thresholds
    p4 = pose.reshape(N320, K, 5)
    # Reference rule on the selected anchors.
    scores = 1 / (1 + np.exp(-person.reshape(-1).astype(np.float64)))
    keep = np.argsort(-scores, kind="stable")[: vis.shape[0]]
    present = 1 / (1 + np.exp(-p4[keep, :, 3].astype(np.float64)))
    visible = 1 / (1 + np.exp(-p4[keep, :, 4].astype(np.float64)))
    expected = np.where(present < present_threshold, 0, np.where(visible >= visible_threshold, 2, 1))
    far = (np.abs(present - present_threshold) > 1e-6) & (np.abs(visible - visible_threshold) > 1e-6)
    np.testing.assert_array_equal(vis[far], expected[far])
    if (present_threshold, visible_threshold) == (0.6, 0.8):
        assert (vis == 0).sum() > (vis_d == 0).sum() and (vis == 2).sum() < (vis_d == 2).sum()
    # decode_keypoints exposes the same class for every anchor.
    anchors, strides = torch_build_anchors(feature_sizes=[(S // s, S // s) for s in STRIDES], strides=list(STRIDES),
                                           device=torch.device("cpu"), dtype=torch.float32)
    dec = decode_keypoints(torch.from_numpy(pose).reshape(1, N320, K, 5), anchors, strides, **th)
    np.testing.assert_array_equal(dec["visibility"][0].numpy()[keep], vis_t)


def _shifted(pose: Pose, shift: np.ndarray, score: float) -> Pose:
    kp = pose.keypoints.copy()
    kp[:, :2] += shift.astype(np.float32)
    return Pose(score=score, keypoints=kp, visibility=pose.visibility.copy(), sigma_px=pose.sigma_px.copy())


def test_oks_nms_iou_threshold_and_scale_factor() -> None:
    from xtx4.runtime import box_iou_matrix, oks_matrix
    from xtx4.runtime.decode import _extent_boxes

    a = _pose(0.9, 0.0)
    # Nearly the same extent box, but the face landmarks displaced (e.g. a mis-localised duplicate):
    shift = np.array([[0, 9], [-8, 9], [8, 9], [-6, 9], [6, 9], [0, 0], [0, 0], [0, 0], [0, 0]], np.float64)
    b = _shifted(a, 2.0 * shift, 0.6)
    oks = oks_matrix([a, b])
    kpts = np.stack([a.keypoints, b.keypoints]).astype(np.float64)
    iou = box_iou_matrix(_extent_boxes(kpts, 0.5, 1.5, 16.0)[1])
    assert oks[0, 1] < 0.5 and iou[0, 1] >= 0.5, (oks, iou)
    assert len(oks_nms([a, b], threshold=0.5)) == 2  # OKS alone keeps the pair
    kept = oks_nms([b, a], threshold=0.5, iou_threshold=0.5)  # OKS OR IoU suppresses it
    assert len(kept) == 1 and kept[0] is a
    # A larger OKS scale (closer to the full-person GT box) raises the OKS of the pair.
    oks_big = oks_matrix([a, b], scale_factor=4.0)
    assert oks_big[0, 1] > oks[0, 1]
    assert len(oks_nms([a, b], threshold=oks[0, 1] + 1e-3, scale_factor=4.0)) == 1
    # A distant person is never IoU-suppressed.
    assert len(oks_nms([a, _pose(0.8, 300.0)], threshold=0.5, iou_threshold=0.1)) == 2
    # The Python API delegates to the same implementation.
    api = pytest.importorskip("xtx4.api")
    arr, sc = np.stack([a.keypoints, b.keypoints]), np.array([0.9, 0.6])
    assert api.oks_nms(arr, sc, threshold=0.5).tolist() == [0, 1]
    assert api.oks_nms(arr, sc, threshold=0.5, iou_threshold=0.5).tolist() == [0]
    assert api.oks_nms(arr, sc, threshold=oks[0, 1] + 1e-3, scale_factor=4.0).tolist() == [0]


def test_manifest_and_runner_carry_decode_thresholds(artefact_dir: Path) -> None:
    decode = {"conf_threshold": 0.3, "present_threshold": 0.6, "visible_threshold": 0.8,
              "oks_nms_iou_threshold": 0.5, "oks_nms_scale_factor": 2.0, "oks_nms_threshold": 0.4}
    manifest = build_manifest(
        model_id="bench-u_b", variant_spec={"name": "u_b"}, strides=[8, 16, 32], levels="P3P4P5", sizes=[320],
        ncnn_files={320: {"param": str(GRAPHS / "u_b_320.ncnn.param"), "bin": str(GRAPHS / "u_b_320.ncnn.bin")}},
        decode=decode, tools={},
    )
    path = write_manifest(artefact_dir / "manifest.json", manifest)
    loaded = load_manifest(path)
    for key, value in decode.items():
        assert loaded["decode"][key] == value
    defaults = build_manifest(model_id="x", variant_spec={}, strides=[8, 16, 32], levels="P3P4P5", sizes=[320],
                              tools={})["decode"]
    assert defaults["present_threshold"] == 0.5 and defaults["visible_threshold"] == 0.5
    assert defaults["oks_nms_iou_threshold"] is None and defaults["oks_nms_scale_factor"] == 1.5
    if not (GRAPHS / "u_b_320.ncnn.param").is_file():
        pytest.skip("bench graphs not exported")
    with NcnnPoseRunner.from_manifest(path, 320) as runner:
        assert (runner.present_threshold, runner.visible_threshold) == (0.6, 0.8)
        assert (runner.oks_nms_iou_threshold, runner.oks_nms_scale_factor, runner.oks_nms_threshold) == (0.5, 2.0, 0.4)
        view, _ = letterbox(_smoke_image(), 320)
        person, pose = runner.forward(view)
        res = runner.detect(_smoke_image(), conf_threshold=0.0, max_detections=20)
        _, _, vis, _ = decode_raw(person, pose, network_size=320, strides=(8, 16, 32), num_keypoints=K,
                                  conf_threshold=0.0, max_detections=20, present_threshold=0.6, visible_threshold=0.8)
        np.testing.assert_array_equal(np.stack([p.visibility for p in res.poses]), vis)
    with NcnnPoseRunner.from_manifest(path, 320, present_threshold=0.5, oks_nms_iou_threshold=None) as runner:
        assert runner.present_threshold == 0.5 and runner.oks_nms_iou_threshold is None


def test_api_and_live_config_plumb_decode_keys(artefact_dir: Path) -> None:
    api = pytest.importorskip("xtx4.api")
    from xtx4.engine.trainer import load_config

    config = load_config()
    params = api.decode_params_from_config(config)
    assert params["present_threshold"] == 0.5 and params["visible_threshold"] == 0.5
    assert params["oks_nms_iou_threshold"] is None and params["oks_nms_scale_factor"] == 1.5
    config["model"]["variant"] = "u_a"
    config["decode"].update({"present_threshold": 0.6, "visible_threshold": 0.8, "oks_nms_iou_threshold": 0.5})
    torch.manual_seed(0)
    model = api.load_model(None, device="cpu", fuse=True, config=config)
    resolved = api._resolve(model, None, None, 256, None)
    assert (resolved["present_threshold"], resolved["visible_threshold"], resolved["oks_nms_iou_threshold"]) == (
        0.6, 0.8, 0.5)
    x = torch.rand(1, 3, 256, 256)
    dets, vis, _ = model.predict(x, conf_threshold=0.0, max_detections=10, present_threshold=0.6, visible_threshold=0.8)
    with torch.no_grad():
        ref = decode_detections(model(x).one2one, conf_threshold=0.0, max_detections=10, present_threshold=0.6,
                                visible_threshold=0.8)
    assert torch.equal(vis[0], ref[1][0]) and torch.equal(dets[0], ref[0][0])

    sys.path.insert(0, str(REPO / "live_test"))
    try:
        from livetest.config import load_config as load_live_config
    finally:
        sys.path.pop(0)
    live = load_live_config(REPO / "live_test" / "config.json")
    assert live.inference.decode_overrides() == {}
    raw = json.loads((REPO / "live_test" / "config.json").read_text(encoding="utf-8"))
    raw["inference"].update({"present_threshold": 0.6, "oks_nms_iou_threshold": None, "oks_nms_scale_factor": 2})
    for net in raw["networks"]:
        for key in ("param", "bin", "manifest"):
            if key in net and not Path(net[key]).is_absolute():
                net[key] = str((REPO / "live_test" / net[key]).resolve())
    (artefact_dir / "live.json").write_text(json.dumps(raw), encoding="utf-8")
    live = load_live_config(artefact_dir / "live.json")
    assert live.inference.decode_overrides() == {"present_threshold": 0.6, "oks_nms_scale_factor": 2.0,
                                                 "oks_nms_iou_threshold": None}
