"""XTX4 live test - camera -> ncnn pose network (xtx4.runtime) -> 9-point skeleton overlay.

Usage:
    python main.py                                  # selection dialog, then live window
    python main.py --network xtx4-u --size 320 --camera 0 --no-dialog
    python main.py --camera video.mp4 --no-dialog   # video file or RTSP/HTTP URL
    python main.py --image photo.jpg [--save out.png]
    python main.py --headless --frames 500 --camera 0          # pipeline benchmark, no window
    python main.py --headless --frames 300 --image photo.jpg   # benchmark on a still image (30 fps source)
    python main.py --list-cameras

No torch: everything model-related comes from ``xtx4.runtime`` one directory up.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import cv2

import livetest  # noqa: F401 - puts the xtx4 repository root on sys.path
from livetest.camera import FrameReader, StillImageReader, list_cameras, source_from_uri
from livetest.config import AppConfig, ConfigError, NetworkConfig, load_config
from livetest.launcher import Selection, load_last, parse_resolution, save_last, select
from livetest.runner import RunnerError, create_runner, shutdown_gpu

logger = logging.getLogger("live_test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live pose test for XTX4 ncnn networks.")
    p.add_argument("--config", default=None, help="Path to config.json (default: next to main.py)")
    p.add_argument("--network", default=None, help="Network id from config.json")
    p.add_argument("--size", type=int, default=None, help="Network input size (e.g. 256/320/384)")
    p.add_argument("--camera", default=None, help="Camera index, video file path or stream URL")
    p.add_argument("--resolution", default=None, help="Requested camera resolution, e.g. 1280x720")
    p.add_argument("--conf", type=float, default=None, help="Detection threshold")
    p.add_argument("--threads", type=int, default=None, help="ncnn threads (default: config inference.num_threads)")
    p.add_argument("--fp16", dest="fp16", action="store_true", default=None, help="ncnn FP16 storage/arithmetic")
    p.add_argument("--fp32", dest="fp16", action="store_false", help="ncnn FP32 arithmetic")
    p.add_argument("--mirror", dest="mirror", action="store_true", default=None, help="Mirror the image")
    p.add_argument("--no-mirror", dest="mirror", action="store_false", help="Do not mirror the image")
    p.add_argument("--vulkan", dest="vulkan", action="store_true", default=None, help="Use the Vulkan GPU")
    p.add_argument("--cpu", dest="vulkan", action="store_false", help="Force CPU inference")
    p.add_argument("--oks-nms", dest="oks_nms", action="store_true", default=None,
                   help="Suppress duplicate detections with OKS-NMS (default: config inference.oks_nms)")
    p.add_argument("--no-dialog", action="store_true", help="Skip the selection dialog (use args / last selection)")
    p.add_argument("--console", action="store_true", help="Use the console menu instead of the dialog")
    p.add_argument("--list-cameras", action="store_true", help="List cameras and exit")
    p.add_argument("--image", default=None, help="Run on an image file instead of a camera")
    p.add_argument("--save", default=None, help="With --image: write the annotated image here")
    p.add_argument("--headless", action="store_true", help="No window: run the threaded pipeline and print stats")
    p.add_argument("--frames", type=int, default=300, help="--headless: processed frames to measure")
    p.add_argument("--warmup", type=int, default=20, help="--headless: warm-up frames (not measured)")
    p.add_argument("--source-fps", type=float, default=None,
                   help="--headless with --image: simulated camera rate (default: config camera.fps)")
    p.add_argument("--json", default=None, help="--headless: also write the stats summary to this JSON file")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args(argv)


def _conf_for(args: argparse.Namespace, network: NetworkConfig, config: AppConfig) -> float:
    if args.conf is not None:
        return float(args.conf)
    return float(network.conf_threshold or config.inference.conf_threshold)


def _apply_overrides(args: argparse.Namespace, config: AppConfig) -> None:
    if args.threads is not None:
        config.inference.num_threads = int(args.threads)
    if args.fp16 is not None:
        config.inference.fp16 = bool(args.fp16)
    if args.oks_nms is not None:
        config.inference.oks_nms = bool(args.oks_nms)


def selection_from_args(args: argparse.Namespace, config: AppConfig) -> Selection:
    """Build a selection from CLI args, falling back to the last session and config."""

    last = load_last()
    ids = [n.id for n in config.networks]
    network_id = args.network or (last.get("network_id") if last.get("network_id") in ids else None) or ids[0]
    network = config.network(network_id)
    size = args.size or (last.get("size") if last.get("network_id") == network_id else None) or network.preferred_size()
    if size is None:
        raise ConfigError(f"Network '{network_id}' has no model files available"
                          + (f" ({network.manifest_error})" if network.manifest_error else ""))

    if args.camera is not None:
        source = source_from_uri(args.camera)
    elif last.get("source_uri"):
        source = source_from_uri(last["source_uri"])
    else:
        cameras = list_cameras(windows_backend=config.camera.windows_backend,
                               max_probe_index=config.camera.max_probe_index)
        keys = [c.key() for c in cameras]
        if not cameras:
            raise RuntimeError("No camera found; pass --camera <index|file|url>")
        source = cameras[keys.index(last["source_key"])] if last.get("source_key") in keys else cameras[0]

    if args.resolution:
        width, height = parse_resolution(args.resolution)
    else:
        width = int(last.get("width", config.camera.width))
        height = int(last.get("height", config.camera.height))

    return Selection(
        network_id=network_id,
        size=int(size),
        source=source,
        width=width,
        height=height,
        conf_threshold=args.conf if args.conf is not None else float(
            last.get("conf_threshold", network.conf_threshold or config.inference.conf_threshold)),
        mirror=args.mirror if args.mirror is not None else bool(last.get("mirror", config.camera.mirror)),
        use_vulkan=args.vulkan if args.vulkan is not None else bool(last.get("use_vulkan", config.inference.use_vulkan)),
    )


def _network_and_size(args: argparse.Namespace, config: AppConfig) -> tuple[NetworkConfig, int]:
    network = config.network(args.network or config.networks[0].id)
    size = args.size or network.preferred_size()
    if size is None:
        raise ConfigError(f"Network '{network.id}' has no model files available"
                          + (f" ({network.manifest_error})" if network.manifest_error else ""))
    return network, int(size)


def run_image(args: argparse.Namespace, config: AppConfig) -> int:
    from livetest.draw import DrawOptions, draw_poses

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Cannot read image: {args.image}", file=sys.stderr)
        return 1
    network, size = _network_and_size(args, config)
    inf = config.inference
    runner = create_runner(network, size, num_threads=inf.num_threads, use_vulkan=bool(args.vulkan), fp16=inf.fp16,
                           decode_overrides=inf.decode_overrides())
    try:
        res = runner.detect(image, conf_threshold=_conf_for(args, network, config), max_detections=inf.max_detections,
                            oks_nms=inf.oks_nms)
    finally:
        runner.close()
    t = res.timings
    print(f"{len(res.poses)} people ({res.raw_candidates} candidates) | pre {t.preprocess_ms:.1f} ms, "
          f"network {t.inference_ms:.1f} ms, post {t.postprocess_ms:.2f} ms")
    for i, pose in enumerate(res.poses):
        vis = "".join(str(int(v)) for v in pose.visibility)
        print(f"  #{i} score={pose.score:.3f} visibility={vis}")
    draw_poses(image, res.poses, network.skeleton,
               DrawOptions(kpt_threshold=inf.kpt_threshold, point_radius=config.display.point_radius,
                           line_thickness=config.display.line_thickness))
    if args.save:
        cv2.imwrite(args.save, image)
        print(f"Saved {args.save}")
    else:
        cv2.namedWindow(config.display.window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(config.display.window_title, image)
        print("Press any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_headless(args: argparse.Namespace, config: AppConfig) -> int:
    """Threaded pipeline benchmark without a window (camera, video or a looped still image)."""

    from xtx4.runtime import run_headless as run_pipeline

    network, size = _network_and_size(args, config)
    inf = config.inference
    if args.image:
        reader = StillImageReader(args.image)
        source_fps = float(args.source_fps if args.source_fps is not None else config.camera.fps)
    else:
        if args.camera is None:
            raise RuntimeError("--headless needs --camera <index|file|url> or --image <file>")
        width, height = parse_resolution(args.resolution) if args.resolution else (config.camera.width,
                                                                                    config.camera.height)
        reader = FrameReader(source_from_uri(args.camera), width=width, height=height, fps=config.camera.fps,
                             mjpg=config.camera.mjpg)
        source_fps = 0.0
    runner = create_runner(network, size, num_threads=inf.num_threads, use_vulkan=bool(args.vulkan), fp16=inf.fp16,
                           decode_overrides=inf.decode_overrides())
    conf = _conf_for(args, network, config)
    mirror = args.mirror if args.mirror is not None else False

    def detect(frame):
        return runner.detect(frame, conf_threshold=conf, max_detections=inf.max_detections, oks_nms=inf.oks_nms,
                             keep_views=False)

    print(f"Headless: {network.id} @{size} | {runner.describe()} | source {reader.source.describe()} "
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
        summary.update({"network": network.id, "size": size, "runner": runner.describe()})
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(f"torch imported: {'torch' in sys.modules}")
    return 0


def run_live(args: argparse.Namespace, config: AppConfig) -> int:
    from livetest.app import LiveApp

    if args.no_dialog:
        selection = selection_from_args(args, config)
    else:
        selection = select(config, console=args.console)
        if selection is None:
            print("Cancelled.")
            return 0
    save_last(selection)

    network = config.network(selection.network_id)
    print(f"Network: {network.id} @{selection.size} ({network.mode}) | source: {selection.source.describe()}")
    try:
        reader = FrameReader(
            selection.source,
            width=selection.width,
            height=selection.height,
            fps=config.camera.fps,
            mjpg=config.camera.mjpg,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Capturing at {reader.frame_size[0]}x{reader.frame_size[1]}. Press 'h' in the window for help.")
    try:
        app = LiveApp(
            config,
            network,
            selection.size,
            reader,
            conf_threshold=selection.conf_threshold,
            mirror=selection.mirror,
            use_vulkan=selection.use_vulkan,
        )
        app.run()
    finally:
        reader.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        config = load_config(args.config)
        _apply_overrides(args, config)
        if args.list_cameras:
            cameras = list_cameras(windows_backend=config.camera.windows_backend,
                                   max_probe_index=config.camera.max_probe_index)
            for cam in cameras:
                print(cam.describe())
            if not cameras:
                print("No camera found.")
            return 0
        if args.headless:
            return run_headless(args, config)
        if args.image:
            return run_image(args, config)
        return run_live(args, config)
    except (ConfigError, RunnerError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        shutdown_gpu()


if __name__ == "__main__":
    sys.exit(main())
