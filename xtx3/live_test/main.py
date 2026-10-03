"""XTX3 live test - camera -> NCNN pose network -> skeleton overlay.

Usage:
    python main.py                                  # selection dialog, then live window
    python main.py --network xtx3-u --size 320 --camera 0 --no-dialog
    python main.py --camera video.mp4 --no-dialog   # video file or RTSP/HTTP URL
    python main.py --image photo.jpg [--save out.png]
    python main.py --list-cameras
"""

from __future__ import annotations

import argparse
import logging
import sys

import cv2

from livetest.camera import FrameGrabber, list_cameras, source_from_uri
from livetest.config import AppConfig, ConfigError, load_config
from livetest.launcher import Selection, load_last, parse_resolution, save_last, select
from livetest.runner import RunnerError, shutdown_gpu

logger = logging.getLogger("live_test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live pose test for XTX3 NCNN networks.")
    p.add_argument("--config", default=None, help="Path to config.json (default: next to main.py)")
    p.add_argument("--network", default=None, help="Network id from config.json")
    p.add_argument("--size", type=int, default=None, help="Network input size (e.g. 256/320/384)")
    p.add_argument("--camera", default=None, help="Camera index, video file path or stream URL")
    p.add_argument("--resolution", default=None, help="Requested camera resolution, e.g. 1280x720")
    p.add_argument("--conf", type=float, default=None, help="Detection threshold")
    p.add_argument("--mirror", dest="mirror", action="store_true", default=None, help="Mirror the image")
    p.add_argument("--no-mirror", dest="mirror", action="store_false", help="Do not mirror the image")
    p.add_argument("--vulkan", dest="vulkan", action="store_true", default=None, help="Use the Vulkan GPU")
    p.add_argument("--cpu", dest="vulkan", action="store_false", help="Force CPU inference")
    p.add_argument("--dedup", dest="dedup", action="store_true", default=None,
                   help="Suppress duplicate detections (default: config inference.dedup)")
    p.add_argument("--no-dialog", action="store_true", help="Skip the selection dialog (use args / last selection)")
    p.add_argument("--console", action="store_true", help="Use the console menu instead of the dialog")
    p.add_argument("--list-cameras", action="store_true", help="List cameras and exit")
    p.add_argument("--image", default=None, help="Run once on an image file instead of a camera")
    p.add_argument("--save", default=None, help="With --image: write the annotated image here")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args()


def selection_from_args(args: argparse.Namespace, config: AppConfig) -> Selection:
    """Build a selection from CLI args, falling back to the last session and config."""

    last = load_last()
    network_id = args.network or last.get("network_id") or config.networks[0].id
    network = config.network(network_id)
    size = args.size or (last.get("size") if last.get("network_id") == network_id else None) or network.preferred_size()
    if size is None:
        raise ConfigError(f"Network '{network_id}' has no model files available")

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


def run_image(args: argparse.Namespace, config: AppConfig) -> int:
    from livetest.draw import DrawOptions, draw_poses
    from livetest.runner import NcnnPoseRunner

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Cannot read image: {args.image}", file=sys.stderr)
        return 1
    network = config.network(args.network or config.networks[0].id)
    size = args.size or network.preferred_size()
    runner = NcnnPoseRunner(network, size, num_threads=config.inference.num_threads,
                            use_vulkan=bool(args.vulkan))
    try:
        res = runner.detect(
            image,
            conf_threshold=args.conf if args.conf is not None else (network.conf_threshold or config.inference.conf_threshold),
            max_detections=config.inference.max_detections,
            oks_match_thr=config.inference.oks_match_thr,
            iou_match_thr=config.inference.iou_match_thr,
            dedup=args.dedup if args.dedup is not None else config.inference.dedup,
        )
    finally:
        runner.close()
    print(f"{len(res.poses)} people, network {res.timings.inference_ms:.1f} ms")
    for i, pose in enumerate(res.poses):
        print(f"  #{i} score={pose.score:.3f} bbox={[round(float(v), 1) for v in pose.bbox]}")
    draw_poses(image, res.poses, network.skeleton,
               DrawOptions(kpt_threshold=config.inference.kpt_threshold,
                           point_radius=config.display.point_radius,
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
        grabber = FrameGrabber(
            selection.source,
            width=selection.width,
            height=selection.height,
            fps=config.camera.fps,
            mjpg=config.camera.mjpg,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Capturing at {grabber.frame_size[0]}x{grabber.frame_size[1]}. Press 'h' in the window for help.")
    try:
        app = LiveApp(
            config,
            network,
            selection.size,
            grabber,
            conf_threshold=selection.conf_threshold,
            mirror=selection.mirror,
            use_vulkan=selection.use_vulkan,
        )
        app.run()
    finally:
        grabber.close()
    return 0


def main() -> int:
    args = parse_args()
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
