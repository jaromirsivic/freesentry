"""XTX4 test app - camera / video / image -> XTX4 pose network (xtx4.runtime) -> 9-point skeleton.

Usage:
    python main.py                                   # test app window (last session's settings)
    python main.py --preset "Pi 5"                   # start with a preset (prefix of its name is enough)
    python main.py --network u_c --size 384 --backend onnxruntime --camera 0
    python main.py --camera video.mp4                # video file or RTSP/HTTP URL
    python main.py --image photo.jpg                 # still image in the app (re-detected on every change)
    python main.py --image photo.jpg --save out.png  # annotate one image, no window
    python main.py --headless --frames 500 --camera 0          # pipeline benchmark, no window
    python main.py --headless --frames 300 --image photo.jpg   # benchmark on a still image (30 fps source)
    python main.py --list-cameras | --list-presets | --list-networks

No torch: everything model-related comes from ``xtx4.runtime`` one directory up.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import cv2

import livetest  # noqa: F401 - puts the xtx4 repository root on sys.path
from livetest.backends import RunnerError, create_runner, shutdown_gpu
from livetest.camera import FrameReader, StillImageReader, list_cameras, source_from_uri
from livetest.config import AppConfig, ConfigError, load_config
from livetest.settings import (
    Settings,
    all_presets,
    apply_preset,
    config_defaults,
    find_preset,
    from_dict,
    load_last,
    validate_for_config,
)

logger = logging.getLogger("live_test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XTX4 test app (live pose test of the exported XTX4 networks).")
    p.add_argument("--config", default=None, help="Path to config.json (default: next to main.py)")
    p.add_argument("--preset", default=None, help="Apply a preset (name or unique prefix; see --list-presets)")
    p.add_argument("--fresh", action="store_true", help="Ignore the last session's settings")
    p.add_argument("--network", default=None, help="Network id from config.json")
    p.add_argument("--size", type=int, default=None, help="Network input size (e.g. 256/320/384)")
    p.add_argument("--backend", choices=["ncnn", "onnxruntime"], default=None,
                   help="ncnn (deployment) or onnxruntime (FP32 reference)")
    p.add_argument("--mode", choices=["whole", "pyramid"], default=None, help="View mode")
    p.add_argument("--camera", default=None, help="Camera index, video file path or stream URL")
    p.add_argument("--resolution", default=None, help="Requested camera resolution, e.g. 1280x720")
    p.add_argument("--conf", type=float, default=None, help="Detection threshold")
    p.add_argument("--threads", type=int, default=None, help="CPU threads (ncnn / ONNX Runtime)")
    p.add_argument("--fp16", dest="fp16", action="store_true", default=None, help="ncnn FP16 storage/arithmetic")
    p.add_argument("--fp32", dest="fp16", action="store_false", help="ncnn FP32 arithmetic")
    p.add_argument("--mirror", dest="mirror", action="store_true", default=None, help="Mirror the image")
    p.add_argument("--no-mirror", dest="mirror", action="store_false", help="Do not mirror the image")
    p.add_argument("--vulkan", dest="vulkan", action="store_true", default=None, help="Use the Vulkan GPU (ncnn)")
    p.add_argument("--cpu", dest="vulkan", action="store_false", help="Force CPU inference")
    p.add_argument("--oks-nms", dest="oks_nms", action="store_true", default=None, help="Enable OKS-NMS")
    p.add_argument("--no-oks-nms", dest="oks_nms", action="store_false", help="Disable OKS-NMS")
    p.add_argument("--list-cameras", action="store_true", help="List cameras and exit")
    p.add_argument("--list-presets", action="store_true", help="List presets and exit")
    p.add_argument("--list-networks", action="store_true", help="List configured networks and exit")
    p.add_argument("--image", default=None, help="Use an image file as the source")
    p.add_argument("--save", default=None, help="With --image: write the annotated image here and exit (no window)")
    p.add_argument("--headless", action="store_true", help="No window: run the threaded pipeline and print stats")
    p.add_argument("--frames", type=int, default=300, help="--headless: processed frames to measure")
    p.add_argument("--warmup", type=int, default=20, help="--headless: warm-up frames (not measured)")
    p.add_argument("--source-fps", type=float, default=None,
                   help="--headless with --image: simulated camera rate (default: config camera.fps)")
    p.add_argument("--json", default=None, help="--headless: also write the stats summary to this JSON file")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args(argv)


def resolve_settings(args: argparse.Namespace, config: AppConfig, *, use_last: bool) -> Settings:
    """Config defaults (+ last session) -> preset -> command line."""

    settings = load_last(config) if use_last and not args.fresh else config_defaults(config)
    if args.preset:
        settings = apply_preset(settings, find_preset(args.preset, all_presets(config)), config)
    overrides: dict = {}
    if args.network is not None:
        config.network(args.network)  # unknown id -> ConfigError
        overrides["network_id"] = args.network
        if args.size is None:
            overrides["size"] = config.network(args.network).preferred_size(args.backend or settings.backend) or 0
    for key, value in (("size", args.size), ("backend", args.backend), ("mode", args.mode),
                       ("conf_threshold", args.conf), ("num_threads", args.threads), ("fp16", args.fp16),
                       ("mirror", args.mirror), ("use_vulkan", args.vulkan), ("oks_nms", args.oks_nms)):
        if value is not None:
            overrides[key] = value
    if args.resolution:
        width, _, height = args.resolution.lower().partition("x")
        overrides["width"], overrides["height"] = int(width), int(height)
    if args.image:
        overrides.update(source_kind="image", source_path=args.image)
    elif args.camera is not None:
        if args.camera.strip().isdigit():
            overrides.update(source_kind="camera", camera_index=int(args.camera), camera_name="")
        else:
            overrides.update(source_kind="video", source_path=args.camera)
    settings = from_dict(overrides, settings)
    requested_size = settings.size
    settings = validate_for_config(settings, config)
    if args.size is not None and settings.size != requested_size:
        network = config.network(settings.network_id)
        raise ConfigError(f"Size {requested_size} is not available for '{network.id}' with {settings.backend} "
                          f"(available: {network.available_sizes(settings.backend)})")
    return settings


def run_image(settings: Settings, config: AppConfig, save: str) -> int:
    from livetest.draw import draw_poses, options_from_settings
    from livetest.gui import pose_to_dict

    image = cv2.imread(settings.source_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Cannot read image: {settings.source_path}", file=sys.stderr)
        return 1
    network = config.network(settings.network_id)
    runner = create_runner(network, settings)
    try:
        res = runner.detect(image, **settings.detect_kwargs())
        description = runner.describe()
    finally:
        runner.close()
    t = res.timings
    print(f"{description}")
    print(f"{len(res.poses)} people ({res.raw_candidates} candidates) | pre {t.preprocess_ms:.1f} ms, "
          f"network {t.inference_ms:.1f} ms, post {t.postprocess_ms:.2f} ms")
    for i, pose in enumerate(res.poses):
        vis = "".join(str(int(v)) for v in pose.visibility)
        print(f"  #{i} score={pose.score:.3f} visibility={vis}")
    draw_poses(image, res.poses, network.skeleton, options_from_settings(settings))
    cv2.imwrite(save, image)
    with open(save + ".json", "w", encoding="utf-8") as handle:
        json.dump({"runner": description, "settings": settings.to_dict(),
                   "poses": [pose_to_dict(p, network.skeleton.keypoints) for p in res.poses]}, handle, indent=2)
    print(f"Saved {save} (+ .json)")
    return 0


def run_headless(args: argparse.Namespace, settings: Settings, config: AppConfig) -> int:
    """Threaded pipeline benchmark without a window (camera, video or a looped still image)."""

    from xtx4.runtime import run_headless as run_pipeline

    network = config.network(settings.network_id)
    if args.image:
        reader = StillImageReader(args.image)
        source_fps = float(args.source_fps if args.source_fps is not None else settings.camera_fps)
    else:
        if args.camera is None:
            raise RuntimeError("--headless needs --camera <index|file|url> or --image <file>")
        reader = FrameReader(source_from_uri(args.camera), width=settings.width, height=settings.height,
                             fps=settings.camera_fps, mjpg=settings.mjpg)
        source_fps = 0.0
    runner = create_runner(network, settings)
    kwargs = dict(settings.detect_kwargs(), keep_views=False)
    mirror = args.mirror if args.mirror is not None else False

    def detect(frame):
        return runner.detect(frame, **kwargs)

    print(f"Headless: {network.id} @{settings.size} | {runner.describe()} | source {reader.source.describe()} "
          f"{reader.frame_size[0]}x{reader.frame_size[1]}" + (f" @{source_fps:.0f} fps" if source_fps else ""))
    try:
        stats = run_pipeline(reader.read, detect, frames=args.frames, warmup=args.warmup,
                             frame_transform=(lambda f: cv2.flip(f, 1)) if mirror else None,
                             max_capture_fps=source_fps, timeout_s=max(60.0, args.frames * 2.0))
    finally:
        reader.close()
        runner.close()
    print(stats.report())
    try:
        import os

        import psutil

        print(f"RSS {psutil.Process(os.getpid()).memory_info().rss / 2**20:.0f} MB")
    except ImportError:
        pass
    if args.json:
        summary = stats.summary()
        summary.update({"network": network.id, "size": settings.size, "runner": runner.describe(),
                        "settings": settings.to_dict()})
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(f"torch imported: {'torch' in sys.modules}")
    return 0


def list_networks(config: AppConfig) -> None:
    for net in config.networks:
        sizes = net.available_sizes()
        onnx = net.available_sizes("onnxruntime")
        state = f"ncnn {sizes or 'missing'}, onnx {onnx or '-'}"
        print(f"{net.id:<14} {state:<40} {net.label}")
        if net.manifest_error:
            print(f"{'':<14} manifest: {net.manifest_error}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        config = load_config(args.config)
        if args.list_cameras:
            cameras = list_cameras(windows_backend=config.camera.windows_backend,
                                   max_probe_index=config.camera.max_probe_index)
            for cam in cameras:
                print(cam.describe())
            if not cameras:
                print("No camera found.")
            return 0
        if args.list_presets:
            for preset in all_presets(config):
                print(f"{preset.name}{'' if preset.builtin else '  (user)'}")
                if preset.description:
                    print(f"    {preset.description}")
            return 0
        if args.list_networks:
            list_networks(config)
            return 0
        interactive = not (args.headless or args.save)
        settings = resolve_settings(args, config, use_last=interactive)
        if args.headless:
            return run_headless(args, settings, config)
        if args.save:
            if not args.image:
                raise ConfigError("--save needs --image")
            return run_image(settings, config, args.save)
        from livetest.gui import run_gui

        return run_gui(config, settings)
    except (ConfigError, RunnerError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        shutdown_gpu()


if __name__ == "__main__":
    sys.exit(main())
