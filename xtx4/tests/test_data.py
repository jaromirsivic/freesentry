"""Tests for the XTX4 data package (scan, augment, dataset, sampler, cache, manifest)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from xtx4.data.annotations import (
    DatasetSpec,
    ImageSample,
    PersonAnnotation,
    scan_split,
)
from xtx4.data.augment import (
    AugmentConfig,
    SourceTargets,
    horizontal_flip,
    mosaic4,
    upper_body_crop,
)
from xtx4.data.cache import build_cache
from xtx4.data.dataset import (
    XTX4PoseDataset,
    build_eval_ground_truth,
    build_targets,
    collate_fn,
    denormalise_sample,
)
from xtx4.data.manifest import build_manifest, check_overlap, load_manifest, write_manifest
from xtx4.data.sampler import MultiScaleBatchSampler
from xtx4.utils import geometry
from xtx4.utils.keypoints import NUM_KEYPOINTS

SMOKE = Path(__file__).resolve().parents[2] / "xtx3" / ".smoke_dataset"
U_JSON = Path(__file__).resolve().parents[1] / "configs" / "u.json"

pytestmark = pytest.mark.skipif(not SMOKE.is_dir(), reason="smoke dataset not available")


@pytest.fixture(scope="module")
def train_samples() -> list[ImageSample]:
    return scan_split(SMOKE / "train")


def _write_image(path: Path, w: int, h: int, value: int = 90) -> None:
    img = np.full((h, w, 3), value, np.uint8)
    cv2.imwrite(str(path), img)


def _person(points: dict[int, tuple[float, float, int]], bbox=None, area=None, iscrowd=0) -> PersonAnnotation:
    kp = np.zeros((NUM_KEYPOINTS, 3), np.float32)
    for k, (x, y, v) in points.items():
        kp[k] = (x, y, v)
    return PersonAnnotation(
        keypoints_norm=kp,
        bbox_xywh_norm=None if bbox is None else np.asarray(bbox, np.float32),
        area_norm=area,
        iscrowd=iscrowd,
    )


# --------------------------------------------------------------------------- annotations


def test_scan_counts_and_visibility(train_samples):
    test_samples = scan_split(SMOKE / "test")
    assert len(train_samples) == 120
    assert len(test_samples) == 40
    paths = [str(s.image_path) for s in train_samples]
    assert paths == sorted(paths)
    persons = [a for s in train_samples for a in s.annotations]
    assert len(persons) > 300
    vals = np.unique(np.concatenate([a.keypoints_norm[:, 2] for a in persons]))
    assert set(vals.tolist()) <= {0.0, 1.0, 2.0}
    assert all(a.num_labeled >= 1 for a in persons)
    assert all(a.area_norm is None for a in persons)  # XTX layout has no area
    kp = np.concatenate([a.keypoints_norm[:, :2] for a in persons])
    assert kp.min() >= 0.0 and kp.max() <= 1.0


def test_absolute_coco_area_crowd_and_unlabeled(tmp_path):
    _write_image(tmp_path / "a.jpg", 200, 100)
    kp17 = [0.0] * 51
    kp17[0:3] = [50, 20, 2]  # nose
    kp17[15:18] = [40, 40, 1]  # left shoulder
    anns = [
        {"id": 1, "image_id": 7, "category_id": 1, "bbox": [30, 10, 40, 80], "area": 1000.0,
         "iscrowd": 0, "num_keypoints": 2, "keypoints": kp17},
        {"id": 2, "image_id": 7, "category_id": 1, "bbox": [100, 10, 60, 60], "area": 2000.0,
         "iscrowd": 1, "num_keypoints": 0, "keypoints": [0] * 51},
        {"id": 3, "image_id": 7, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0,
         "num_keypoints": 0, "keypoints": [0] * 51},
    ]
    data = {"images": [{"id": 7, "file_name": "a.jpg", "width": 200, "height": 100}], "annotations": anns}
    (tmp_path / "ann.json").write_text(json.dumps(data))
    samples = scan_split(tmp_path, DatasetSpec("coco", "coco17", None))
    assert len(samples) == 1 and samples[0].dataset == "coco"
    persons = samples[0].annotations
    assert len(persons) == 2  # unlabeled non-crowd skipped
    p, crowd = persons
    np.testing.assert_allclose(p.keypoints_norm[0], [0.25, 0.2, 2], atol=1e-6)
    np.testing.assert_allclose(p.keypoints_norm[5], [0.2, 0.4, 1], atol=1e-6)
    assert p.area_norm == pytest.approx(1000.0 / 20000.0)
    np.testing.assert_allclose(p.bbox_xywh_norm, [0.15, 0.1, 0.2, 0.8], atol=1e-6)
    assert crowd.iscrowd == 1 and (crowd.keypoints_norm[:, 2] == 0).all()
    assert crowd.bbox_xywh_norm is not None
    with_ignore = scan_split(tmp_path, DatasetSpec("coco", "coco17", None), unlabeled_as_ignore=True)
    assert [a.iscrowd for a in with_ignore[0].annotations] == [0, 1, 1]


def _write_partial_json(path: Path, n_points: int, labeled: dict[int, tuple[float, float]]) -> None:
    kp = [0.0] * (3 * n_points)
    for i, (x, y) in labeled.items():
        kp[3 * i : 3 * i + 3] = [x, y, 2]
    data = {
        "images": [{"id": 1, "file_name": "p.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 80, 80],
                         "iscrowd": 0, "keypoints": kp}],
    }
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "fmt,n_points,lsho,rsho,lhip,rhip",
    [("crowdpose", 14, 0, 1, 6, 7), ("aic", 14, 3, 0, 9, 6)],
)
def test_partial_formats_face_unknown(tmp_path, fmt, n_points, lsho, rsho, lhip, rhip):
    _write_image(tmp_path / "p.jpg", 100, 100)
    labeled = {lsho: (30, 30), rsho: (70, 31), lhip: (35, 70), rhip: (65, 71), 12: (50, 5), 13: (50, 25)}
    _write_partial_json(tmp_path / "a.json", n_points, labeled)
    samples = scan_split(tmp_path, DatasetSpec(fmt, fmt, None))
    kp = samples[0].annotations[0].keypoints_norm
    assert (kp[:5, 2] == -1).all()
    np.testing.assert_allclose(kp[5], [0.30, 0.30, 2])
    np.testing.assert_allclose(kp[6], [0.70, 0.31, 2])
    np.testing.assert_allclose(kp[7], [0.35, 0.70, 2])
    np.testing.assert_allclose(kp[8], [0.65, 0.71, 2])


def test_mpii_mmpose_list(tmp_path):
    (tmp_path / "images").mkdir()
    _write_image(tmp_path / "images" / "m.jpg", 100, 50)
    joints = [[0.0, 0.0]] * 16
    joints = [list(j) for j in joints]
    vis = [0] * 16
    for i, xy in {2: (60, 40), 3: (40, 40), 12: (62, 20), 13: (38, 20)}.items():
        joints[i] = list(xy)
        vis[i] = 1
    (tmp_path / "mpii.json").write_text(json.dumps([{"image": "m.jpg", "joints": joints, "joints_vis": vis}]))
    samples = scan_split(tmp_path, DatasetSpec("mpii", "mpii", None))
    kp = samples[0].annotations[0].keypoints_norm
    assert (kp[:5, 2] == -1).all()
    np.testing.assert_allclose(kp[5], [0.38, 0.4, 2])  # left shoulder = MPII 13
    np.testing.assert_allclose(kp[8], [0.60, 0.8, 2])  # right hip = MPII 2


# --------------------------------------------------------------------------- augment


def test_augment_config_from_u_json():
    cfg = AugmentConfig.from_dict(json.loads(U_JSON.read_text())["data"]["augment"])
    assert cfg.scale_min == 0.5 and cfg.scale_max == 1.5 and cfg.upper_body_crop == 0.3
    legacy = AugmentConfig.from_dict({"scale_jitter": 0.25})
    assert legacy.scale_min == 0.75 and legacy.scale_max == 1.25


def test_flip_swaps_pairs_and_keeps_v():
    kp = np.zeros((1, NUM_KEYPOINTS, 3))
    kp[0, :, 0] = np.arange(NUM_KEYPOINTS) * 10 + 5
    kp[0, :, 1] = 7
    kp[0, :, 2] = [2, 1, 2, 0, -1, 2, 1, 1, 2]
    t = SourceTargets(kp, np.array([[10.0, 0, 30, 20]]), np.array([np.nan]), np.zeros(1, bool))
    img = np.zeros((20, 100, 3), np.uint8)
    img[:, :10] = 255
    out_img, out, affine = horizontal_flip(img, t)
    assert out_img[:, -10:].min() == 255
    expected_src = [0, 2, 1, 4, 3, 6, 5, 8, 7]
    np.testing.assert_array_equal(out.keypoints[0, :, 2], kp[0, expected_src, 2])
    for k, src in enumerate(expected_src):
        if kp[0, src, 2] > 0:
            assert out.keypoints[0, k, 0] == pytest.approx(100 - kp[0, src, 0])
            back = geometry.apply_to_points(affine, out.keypoints[0, k : k + 1, :2])
            np.testing.assert_allclose(back[0], kp[0, src, :2])
    np.testing.assert_allclose(out.boxes[0], [70, 0, 90, 20])


def test_mosaic_keeps_aspect_and_maps_points():
    imgs = [np.full((100, 200, 3), 200, np.uint8)] + [np.full((50, 50, 3), 100, np.uint8)] * 3
    kp = np.zeros((1, NUM_KEYPOINTS, 3))
    kp[0, 0] = (50, 25, 2)
    kp[0, 1] = (0, 0, -1)
    tgts = [SourceTargets(kp, np.array([[0.0, 0, 100, 50]]), np.array([400.0]), np.zeros(1, bool))]
    tgts += [SourceTargets.empty()] * 3
    canvas, out, tiles = mosaic4(imgs, tgts, output_size=256)
    tile = tiles[0]
    assert (tile.width, tile.height) == (128, 64)
    assert tile.width / tile.height == pytest.approx(2.0)
    region = canvas[:128, :128, 0]
    ys, xs = np.nonzero(region == 200)
    assert (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1) == (0, 128, 32, 96)
    np.testing.assert_allclose(out.keypoints[0, 0], [32.0, 48.0, 2.0])
    assert out.keypoints[0, 1, 2] == -1
    assert out.area[0] == pytest.approx(400 * 0.64 * 0.64)
    np.testing.assert_allclose(out.boxes[0], [0, 32, 64, 64])


def test_upper_body_crop_cuts_below_hips_or_shoulders():
    rng = np.random.default_rng(3)
    kp = np.zeros((1, NUM_KEYPOINTS, 3))
    kp[0, 0] = (100, 40, 2)
    kp[0, 5] = (80, 80, 2)
    kp[0, 6] = (120, 80, 2)
    kp[0, 7] = (85, 150, 2)
    kp[0, 8] = (115, 150, 1)
    t = SourceTargets(kp, np.array([[60.0, 20, 140, 390]]), np.array([np.nan]), np.zeros(1, bool))
    img = np.zeros((400, 200, 3), np.uint8)
    for _ in range(30):
        cropped, out, affine = upper_body_crop(img, t, rng)
        h, w = cropped.shape[:2]
        assert h * w >= 0.25 * 400 * 200
        assert h < 400  # something below the person was removed
        assert out.keypoints[0, 5, 2] == 2 and out.keypoints[0, 6, 2] == 2
        lab = out.keypoints[0, :, 2] > 0
        assert (out.keypoints[0, lab, 1] < h).all()
        back = geometry.apply_to_points(affine, out.keypoints[0, lab, :2])
        np.testing.assert_allclose(back, kp[0, lab, :2])
        hips_cut = out.keypoints[0, 7:9, 2] == 0
        assert hips_cut.all() or not hips_cut.any()


# --------------------------------------------------------------------------- dataset


def test_head_inside_box_centre_outside_is_target(tmp_path):
    path = tmp_path / "head.png"
    _write_image(path, 200, 200)
    person = _person({0: (0.5, 0.95, 2), 1: (0.48, 0.94, 2), 2: (0.52, 0.94, 1)}, bbox=[0.4, 0.9, 0.2, 1.2])
    sample = ImageSample(image_path=path, width=200, height=200, annotations=[person])
    ds = XTX4PoseDataset([sample], [256], None, False)
    _, targets, meta = ds[(0, 256)]
    assert targets["keypoints"].shape == (1, NUM_KEYPOINTS, 3)
    box = geometry.apply_to_boxes_xyxy(geometry.invert_affine(meta.transform), np.array([[80.0, 180, 120, 420]]))
    assert (box[0, 1] + box[0, 3]) / 2 > 256  # full-box centre is outside the view
    np.testing.assert_allclose(targets["keypoints"][0, 0].numpy(), [128.0, 218.0, 2.0], atol=1e-4)
    assert torch.isnan(targets["area"]).all()

    # Direct: same rule on build_targets with the head pushed partially out of the view.
    kp = np.zeros((1, NUM_KEYPOINTS, 3))
    kp[0, 0] = (50, 250, 2)
    kp[0, 5] = (50, 300, 2)  # leaves the view
    t = SourceTargets(kp, np.array([[0.0, 240, 100, 600]]), np.array([5000.0]), np.zeros(1, bool))
    out = build_targets(t, forward=geometry.identity_affine(), size=256)
    assert out["keypoints"].shape[0] == 1
    assert out["keypoints"][0, 5, 2] == 0 and out["keypoints"][0, 0, 2] == 2
    assert build_targets(t, forward=geometry.identity_affine(), size=256, min_labeled_keypoints=2)["keypoints"].shape[0] == 0


@pytest.mark.parametrize("size", [256, 384])
def test_getitem_shapes_with_augmentation(train_samples, size):
    ds = XTX4PoseDataset(train_samples[:24], [256, 320, 384], AugmentConfig(mosaic=0.5), True, seed=1)
    ds.reseed(123)
    total = 0
    for i in range(24):
        img, tg, meta = ds[(i, size)]
        assert img.shape == (3, size, size) and img.dtype == torch.float32
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
        m = tg["keypoints"].shape[0]
        assert tg["area"].shape == (m,) and tg["boxes"].shape == (m, 4) and tg["iscrowd"].shape == (m,)
        kp = tg["keypoints"]
        lab = kp[..., 2] > 0
        assert ((kp[..., 0][lab] >= 0) & (kp[..., 0][lab] <= size)).all()
        assert ((kp[..., 1][lab] >= 0) & (kp[..., 1][lab] <= size)).all()
        assert set(torch.unique(kp[..., 2]).tolist()) <= {-1.0, 0.0, 1.0, 2.0}
        assert (lab.sum(1) >= 1).all()
        assert meta.size == size and meta.transform.shape == (2, 3)
        total += m
    assert total > 0


@pytest.mark.parametrize("augment", [False, True])
def test_transform_round_trip(train_samples, augment):
    cfg = AugmentConfig(mosaic=0.0, upper_body_crop=0.5, fliplr=0.5)
    ds = XTX4PoseDataset(train_samples[:16], [320], cfg, augment, mosaic=False)
    ds.reseed(7)
    for i in range(16):
        _, tg, meta = ds[(i, 320)]
        sample = train_samples[i]
        _, src_kp = denormalise_sample(sample, width=meta.orig_size[0], height=meta.orig_size[1])
        src_kp = src_kp[[a.num_labeled > 0 for a in sample.annotations]]
        kp = tg["keypoints"].numpy()
        if not augment:
            assert kp.shape[0] == src_kp.shape[0]
        for p in range(kp.shape[0]):
            lab = kp[p, :, 2] > 0
            back = geometry.apply_to_points(meta.transform, kp[p, lab, :2].astype(np.float64))
            # Match to the closest source person (flip may permute L/R indices).
            dists = []
            for q in range(src_kp.shape[0]):
                cand = src_kp[q][:, :2]
                if augment:
                    errs = [np.abs(cand[k] - b).max() if src_kp[q, k, 2] > 0 else np.inf
                            for k, b in zip(np.nonzero(lab)[0], back)]
                    flip = [0, 2, 1, 4, 3, 6, 5, 8, 7]
                    errs_f = [np.abs(cand[flip[k]] - b).max() if src_kp[q, flip[k], 2] > 0 else np.inf
                              for k, b in zip(np.nonzero(lab)[0], back)]
                    dists.append(min(max(errs), max(errs_f)))
                else:
                    dists.append(np.abs(cand[lab] - back).max())
            assert min(dists) <= 0.5, (i, p, min(dists))
        if not augment:
            assert not meta.mosaic


def test_half_pixel_resize(tmp_path):
    # 16x16 image, a single bright 2x2 block at source pixels x 6..7, y 10..11.
    img = np.zeros((16, 16, 3), np.uint8)
    img[10:12, 6:8] = 255
    path = tmp_path / "checker.png"
    cv2.imwrite(str(path), img)
    person = _person({0: (7.0 / 16, 11.0 / 16, 2)})  # block centre in continuous px = (7, 11)
    ds = XTX4PoseDataset([ImageSample(path, 16, 16, [person])], [8], None, False)
    view, tg, meta = ds[(0, 8)]
    v = (view[0].numpy() * 255).round()
    assert v[5, 3] == 255 and v.sum() == 255  # exactly one network pixel lit
    np.testing.assert_allclose(tg["keypoints"][0, 0, :2].numpy(), [3.5, 5.5])  # its pixel centre
    np.testing.assert_allclose(geometry.apply_to_points(meta.transform, np.array([[3.5, 5.5]]))[0], [7.0, 11.0])


def test_collate_and_loader(train_samples):
    ds = XTX4PoseDataset(train_samples[:16], [256, 320], AugmentConfig(), True)
    sampler = MultiScaleBatchSampler(len(ds), 4, [256, 320], [0.5, 0.5], seed=0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
    batches = list(loader)
    assert len(batches) == 4
    for images, targets, metas in batches:
        assert images.ndim == 4 and images.shape[0] == 4 and images.shape[-1] in (256, 320)
        assert len(targets) == 4 and len(metas) == 4
        assert all(m.size == images.shape[-1] for m in metas)


def test_eval_ground_truth(train_samples):
    images, gts = build_eval_ground_truth(train_samples, 3)
    assert len(images) == len(gts) == 3
    for img, gt, s in zip(images, gts, train_samples):
        g = len(s.annotations)
        assert gt["keypoints"].shape == (g, NUM_KEYPOINTS, 3)
        assert gt["areas"].shape == (g,) and (gt["areas"] > 0).all()
        assert gt["iscrowd"].dtype == bool
        assert gt["keypoints"][..., 0].max() <= img.shape[1]
    paths, _ = build_eval_ground_truth(train_samples, 2, load_images=False)
    assert isinstance(paths[0], str)


# --------------------------------------------------------------------------- sampler


def test_sampler_one_size_per_batch_and_coverage():
    sampler = MultiScaleBatchSampler(103, 8, [256, 320, 384], [0.2, 0.5, 0.3], seed=5, drop_last=False)
    assert len(sampler) == 13
    batches = list(sampler)
    assert len(batches) == 13
    seen = []
    for b in batches:
        assert len({s for _, s in b}) == 1
        seen.extend(i for i, _ in b)
    assert sorted(seen) == list(range(103))
    assert list(MultiScaleBatchSampler(103, 8, [256, 320, 384], [0.2, 0.5, 0.3], seed=5, drop_last=False)) == batches
    sampler.set_epoch(1)
    assert list(sampler) != batches
    dl = MultiScaleBatchSampler(103, 8, [320], seed=0)
    assert len(dl) == 12 and len(list(dl)) == 12
    w = np.ones(103)
    w[:10] = 2.0
    weighted = MultiScaleBatchSampler(103, 8, [320], seed=0, weights=w)
    assert len(list(weighted)) == 12


# --------------------------------------------------------------------------- cache / manifest


def test_cache_round_trip(tmp_path, train_samples):
    subset = train_samples[:6]
    cached = build_cache(subset, tmp_path / "cache", max_side=320, num_threads=4)
    again = build_cache(subset, tmp_path / "cache", max_side=320, num_threads=4)
    assert [s.image_path for s in cached] == [s.image_path for s in again]
    for orig, new in zip(subset, cached):
        assert new.image_path != orig.image_path and new.image_path.is_file()
        img = cv2.imread(str(new.image_path))
        assert (img.shape[1], img.shape[0]) == (new.width, new.height)
        assert max(new.width, new.height) <= 320
        assert new.width / new.height == pytest.approx(orig.width / orig.height, rel=0.02)
        assert new.annotations is orig.annotations or all(
            np.array_equal(a.keypoints_norm, b.keypoints_norm) for a, b in zip(orig.annotations, new.annotations)
        )
        _, kp_o = denormalise_sample(orig, width=orig.width, height=orig.height)
        _, kp_n = denormalise_sample(new, width=new.width, height=new.height)
        scale = new.width / orig.width
        np.testing.assert_allclose(kp_n[..., :2], kp_o[..., :2] * scale, atol=1.0)
    ds = XTX4PoseDataset(cached, [256], None, False)
    _, tg, _ = ds[(0, 256)]
    assert tg["keypoints"].shape[0] == sum(a.num_labeled > 0 for a in cached[0].annotations)


def test_manifest_and_overlap(tmp_path, train_samples):
    test_samples = scan_split(SMOKE / "test")
    sidecar = tmp_path / "sha1cache.json"
    ma = build_manifest(train_samples[:20], "train", cache_path=sidecar)
    mb = build_manifest(test_samples[:20], "test", cache_path=sidecar)
    assert ma["num_images"] == 20 and all(len(e["sha1"]) == 40 for e in ma["images"])
    assert check_overlap(ma, mb) == []
    dup = build_manifest(train_samples[:3], "dup", cache_path=sidecar)
    assert len(check_overlap(ma, dup)) == 3
    write_manifest(ma, tmp_path / "m.json")
    assert load_manifest(tmp_path / "m.json")["manifest_sha1"] == ma["manifest_sha1"]
    assert len(json.loads(sidecar.read_text())) >= 40
