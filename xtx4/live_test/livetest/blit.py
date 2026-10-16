"""Put a BGR frame into a Tk widget, scaled to fit (letterboxed, centred).

Tk's photo images are slow for video: converting a ~2 Mpx frame takes 35-50 ms
on Windows (``ImageTk.PhotoImage.paste``), which would cap the display at
~15-20 fps. On Windows the frame is therefore written straight into the widget's
window with GDI ``SetDIBitsToDevice`` (~3-5 ms); elsewhere Pillow's ``ImageTk``
is used.
"""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk

import cv2
import numpy as np

def fit(frame_w: int, frame_h: int, area_w: int, area_h: int) -> tuple[int, int, int, int, float]:
    """``(x0, y0, w, h, scale)`` of the frame letterboxed into the area."""

    scale = min(area_w / frame_w, area_h / frame_h)
    w, h = max(1, int(frame_w * scale)), max(1, int(frame_h * scale))
    return (area_w - w) // 2, (area_h - h) // 2, w, h, scale


def _resized(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    if (frame.shape[1], frame.shape[0]) == (w, h):
        return frame
    interp = cv2.INTER_AREA if w < frame.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(frame, (w, h), interpolation=interp)


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32), ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32),
    ]


class Win32Blitter:
    """GDI blit into the widget's own window (the widget must not draw anything itself)."""

    def __init__(self, widget: tk.Widget) -> None:
        self.widget = widget
        self._user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        self._gdi32 = ctypes.windll.gdi32  # type: ignore[attr-defined]
        self._user32.GetDC.restype = ctypes.c_void_p
        self._user32.GetDC.argtypes = [ctypes.c_void_p]
        self._user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._gdi32.SetDIBitsToDevice.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int,
            ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
        ]
        self._user32.InvalidateRect.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self._rect: tuple[int, int, int, int] | None = None

    def show(self, frame: np.ndarray) -> tuple[int, int, float]:
        """Blit only the image rectangle; the widget's own background paints the borders."""

        area_w, area_h = max(1, self.widget.winfo_width()), max(1, self.widget.winfo_height())
        x0, y0, w, h, scale = fit(frame.shape[1], frame.shape[0], area_w, area_h)
        hwnd = self.widget.winfo_id()
        if self._rect is not None and self._rect != (x0, y0, w, h):
            # Smaller image than before: let Tk repaint the background (it sends <Expose> -> redraw).
            self._user32.InvalidateRect(hwnd, None, 1)
        self._rect = (x0, y0, w, h)
        bgra = cv2.cvtColor(_resized(frame, w, h), cv2.COLOR_BGR2BGRA)
        header = _BitmapInfoHeader(ctypes.sizeof(_BitmapInfoHeader), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        hdc = self._user32.GetDC(hwnd)
        try:
            self._gdi32.SetDIBitsToDevice(hdc, x0, y0, w, h, 0, 0, 0, h, bgra.ctypes.data, ctypes.byref(header), 0)
        finally:
            self._user32.ReleaseDC(hwnd, hdc)
        return x0, y0, scale

    def clear(self) -> None:
        self._rect = None


class PilBlitter:
    """Portable fallback: Pillow ``ImageTk`` image on a Tk canvas."""

    def __init__(self, canvas: tk.Canvas) -> None:
        from PIL import ImageTk  # noqa: F401 - fail early when Pillow is missing

        self.canvas = canvas
        self._photo = None
        self._item: int | None = None

    def show(self, frame: np.ndarray) -> tuple[int, int, float]:
        from PIL import Image, ImageTk

        area_w, area_h = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        x0, y0, w, h, scale = fit(frame.shape[1], frame.shape[0], area_w, area_h)
        image = Image.fromarray(cv2.cvtColor(_resized(frame, w, h), cv2.COLOR_BGR2RGB))
        if self._photo is not None and (self._photo.width(), self._photo.height()) == (w, h):
            self._photo.paste(image)
        else:
            self._photo = ImageTk.PhotoImage(image)
        if self._item is None:
            self._item = self.canvas.create_image(x0, y0, anchor="nw", image=self._photo)
        else:
            self.canvas.itemconfigure(self._item, image=self._photo)
            self.canvas.coords(self._item, x0, y0)
        return x0, y0, scale

    def clear(self) -> None:
        self._item = None
        self._photo = None


def make_blitter(canvas: tk.Canvas):
    if sys.platform == "win32":
        try:
            return Win32Blitter(canvas)
        except (AttributeError, OSError):
            pass
    return PilBlitter(canvas)
