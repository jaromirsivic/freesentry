"""Start-up selection: network, input size and video source.

A small tkinter dialog is used when available; otherwise a numbered console menu.
The dialog is fully destroyed before the OpenCV window opens (on macOS both
toolkits need the main thread and must not run at the same time).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from .camera import CameraSource, list_cameras, source_from_uri
from .config import AppConfig

logger = logging.getLogger(__name__)

LAST_SELECTION_PATH = Path(__file__).resolve().parent.parent / ".last_selection.json"
RESOLUTIONS = ["640x480", "1280x720", "1920x1080"]


@dataclass(slots=True)
class Selection:
    network_id: str
    size: int
    source: CameraSource
    width: int
    height: int
    conf_threshold: float
    mirror: bool
    use_vulkan: bool


def parse_resolution(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


# ------------------------------------------------------------- persistence
def load_last() -> dict:
    try:
        return json.loads(LAST_SELECTION_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_last(sel: Selection) -> None:
    data = asdict(sel)
    data["source_key"] = sel.source.key()
    data["source_uri"] = sel.source.uri
    data.pop("source", None)
    try:
        LAST_SELECTION_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not save last selection: %s", exc)


# --------------------------------------------------------------- helpers
def _network_choices(config: AppConfig) -> list[tuple[str, str, list[int]]]:
    """``(id, display label, available sizes)`` for every configured network."""

    out = []
    for net in config.networks:
        sizes = net.available_sizes()
        label = f"{net.label}" if sizes else f"{net.label}  [model files missing]"
        out.append((net.id, label, sizes))
    return out


def _default_size(config: AppConfig, network_id: str, last: dict) -> int | None:
    net = config.network(network_id)
    sizes = net.available_sizes()
    if last.get("network_id") == network_id and last.get("size") in sizes:
        return int(last["size"])
    return net.preferred_size()


# ------------------------------------------------------------ tk dialog
def select_with_dialog(config: AppConfig) -> Selection | None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    _enable_windows_dpi_awareness()
    last = load_last()
    networks = _network_choices(config)
    cameras: list[CameraSource] = []

    root = tk.Tk()
    root.title("XTX4 live test")
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=14)
    frm.grid(sticky="nsew")
    result: dict[str, Selection] = {}

    def row_label(text: str, row: int) -> None:
        ttk.Label(frm, text=text).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)

    # Network
    row_label("Neural network", 0)
    net_var = tk.StringVar()
    net_box = ttk.Combobox(frm, textvariable=net_var, state="readonly", width=48,
                           values=[label for _, label, _ in networks])
    net_box.grid(row=0, column=1, columnspan=2, sticky="we", pady=4)

    row_label("Input size", 1)
    size_var = tk.StringVar()
    size_box = ttk.Combobox(frm, textvariable=size_var, state="readonly", width=10)
    size_box.grid(row=1, column=1, sticky="w", pady=4)
    mode_label = ttk.Label(frm, foreground="#666")
    mode_label.grid(row=1, column=2, sticky="w")

    # Source
    row_label("Camera", 2)
    cam_var = tk.StringVar()
    cam_box = ttk.Combobox(frm, textvariable=cam_var, state="readonly", width=48)
    cam_box.grid(row=2, column=1, sticky="we", pady=4)
    ttk.Button(frm, text="Refresh", command=lambda: refresh_cameras()).grid(row=2, column=2, sticky="w", padx=(6, 0))

    row_label("or file / URL", 3)
    uri_var = tk.StringVar(value=last.get("source_uri") or "")
    ttk.Entry(frm, textvariable=uri_var, width=50).grid(row=3, column=1, sticky="we", pady=4)

    def browse() -> None:
        path = filedialog.askopenfilename(
            parent=root, title="Select a video file",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov *.webm"), ("All files", "*.*")],
        )
        if path:
            uri_var.set(path)

    ttk.Button(frm, text="Browse…", command=browse).grid(row=3, column=2, sticky="w", padx=(6, 0))

    row_label("Camera resolution", 4)
    default_res = f"{last.get('width', config.camera.width)}x{last.get('height', config.camera.height)}"
    res_values = RESOLUTIONS if default_res in RESOLUTIONS else [default_res, *RESOLUTIONS]
    res_var = tk.StringVar(value=default_res)
    ttk.Combobox(frm, textvariable=res_var, values=res_values, width=12).grid(row=4, column=1, sticky="w", pady=4)

    row_label("Detection threshold", 5)
    conf_var = tk.DoubleVar(value=float(last.get("conf_threshold", config.inference.conf_threshold)))
    conf_text = ttk.Label(frm, width=5)
    conf_text.grid(row=5, column=2, sticky="w")

    def on_conf(*_args) -> None:
        conf_text.configure(text=f"{conf_var.get():.2f}")

    ttk.Scale(frm, from_=0.05, to=0.95, variable=conf_var, command=on_conf).grid(row=5, column=1, sticky="we", pady=4)
    on_conf()

    mirror_var = tk.BooleanVar(value=bool(last.get("mirror", config.camera.mirror)))
    vulkan_var = tk.BooleanVar(value=bool(last.get("use_vulkan", config.inference.use_vulkan)))
    opts = ttk.Frame(frm)
    opts.grid(row=6, column=1, columnspan=2, sticky="w", pady=(4, 8))
    ttk.Checkbutton(opts, text="Mirror image", variable=mirror_var).pack(side="left", padx=(0, 16))
    ttk.Checkbutton(opts, text="Use Vulkan GPU (ncnn)", variable=vulkan_var).pack(side="left")

    buttons = ttk.Frame(frm)
    buttons.grid(row=7, column=0, columnspan=3, sticky="e")
    start_btn = ttk.Button(buttons, text="Start", command=lambda: on_start())
    start_btn.pack(side="right")
    ttk.Button(buttons, text="Quit", command=root.destroy).pack(side="right", padx=(0, 8))

    def current_network_index() -> int:
        return net_box.current()

    def on_network(*_args) -> None:
        idx = current_network_index()
        if idx < 0:
            return
        net_id, _, sizes = networks[idx]
        size_box.configure(values=[str(s) for s in sizes])
        default = _default_size(config, net_id, last)
        size_var.set(str(default) if default is not None else "")
        net = config.network(net_id)
        mode_label.configure(text=f"mode: {net.mode}" + (f" | {net.manifest_data['model_id']}" if net.manifest_data else ""))
        start_btn.state(["!disabled"] if sizes else ["disabled"])

    def refresh_cameras() -> None:
        nonlocal cameras
        root.config(cursor="watch")
        root.update_idletasks()
        try:
            cameras = list_cameras(
                windows_backend=config.camera.windows_backend, max_probe_index=config.camera.max_probe_index
            )
        finally:
            root.config(cursor="")
        cam_box.configure(values=[c.describe() for c in cameras] or ["(no camera found)"])
        keys = [c.key() for c in cameras]
        cam_box.current(keys.index(last["source_key"]) if last.get("source_key") in keys else 0)

    def on_start() -> None:
        idx = current_network_index()
        if idx < 0 or not size_var.get():
            messagebox.showerror("XTX4 live test", "Select a network with available model files.", parent=root)
            return
        uri = uri_var.get().strip()
        if uri:
            source = source_from_uri(uri)
        elif cameras and cam_box.current() >= 0:
            source = cameras[cam_box.current()]
        else:
            messagebox.showerror("XTX4 live test", "No camera selected and no file / URL given.", parent=root)
            return
        try:
            width, height = parse_resolution(res_var.get())
        except ValueError:
            messagebox.showerror("XTX4 live test", "Resolution must look like 1280x720.", parent=root)
            return
        result["selection"] = Selection(
            network_id=networks[idx][0],
            size=int(size_var.get()),
            source=source,
            width=width,
            height=height,
            conf_threshold=round(float(conf_var.get()), 2),
            mirror=bool(mirror_var.get()),
            use_vulkan=bool(vulkan_var.get()),
        )
        root.destroy()

    net_box.bind("<<ComboboxSelected>>", on_network)
    ids = [n[0] for n in networks]
    net_box.current(ids.index(last["network_id"]) if last.get("network_id") in ids else 0)
    on_network()
    refresh_cameras()
    root.bind("<Return>", lambda _e: on_start())
    root.bind("<Escape>", lambda _e: root.destroy())
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return result.get("selection")


def _enable_windows_dpi_awareness() -> None:
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - not Windows / not supported
        pass


# --------------------------------------------------------- console menu
def _ask_choice(prompt: str, options: list[str], default: int = 0) -> int:
    for i, text in enumerate(options):
        print(f"  {i + 1}) {text}")
    while True:
        raw = input(f"{prompt} [{default + 1}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("Invalid choice, try again.")


def select_in_console(config: AppConfig) -> Selection | None:
    last = load_last()
    networks = [n for n in _network_choices(config) if n[2]]
    if not networks:
        print("No configured network has its model files available.")
        return None
    ids = [n[0] for n in networks]
    print("\nNeural network:")
    idx = _ask_choice("Network", [n[1] for n in networks], ids.index(last["network_id"]) if last.get("network_id") in ids else 0)
    net_id, _, sizes = networks[idx]
    default_size = _default_size(config, net_id, last)
    print("\nInput size:")
    size = sizes[_ask_choice("Size", [str(s) for s in sizes], sizes.index(default_size) if default_size in sizes else 0)]

    print("\nSearching for cameras...")
    cameras = list_cameras(windows_backend=config.camera.windows_backend, max_probe_index=config.camera.max_probe_index)
    options = [c.describe() for c in cameras] + ["Video file / URL..."]
    keys = [c.key() for c in cameras]
    default_cam = keys.index(last["source_key"]) if last.get("source_key") in keys else 0
    choice = _ask_choice("Source", options, default_cam)
    if choice == len(cameras):
        uri = input("Path or URL: ").strip()
        if not uri:
            return None
        source = source_from_uri(uri)
    else:
        source = cameras[choice]

    width, height = config.camera.width, config.camera.height
    return Selection(
        network_id=net_id,
        size=size,
        source=source,
        width=width,
        height=height,
        conf_threshold=float(last.get("conf_threshold", config.inference.conf_threshold)),
        mirror=bool(last.get("mirror", config.camera.mirror)),
        use_vulkan=bool(last.get("use_vulkan", config.inference.use_vulkan)),
    )


def select(config: AppConfig, *, console: bool = False) -> Selection | None:
    if not console:
        try:
            return select_with_dialog(config)
        except ImportError:
            logger.info("tkinter is not available; falling back to the console menu")
        except Exception as exc:  # noqa: BLE001 - e.g. no display ($DISPLAY unset)
            logger.warning("Selection dialog failed (%s); falling back to the console menu", exc)
    return select_in_console(config)
