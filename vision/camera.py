from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    flip_horizontal: bool = True


class Camera:
    """Threaded camera wrapper for low-latency frame capture.

    A dedicated reader thread continuously grabs frames in the background.
    ``read()`` blocks until a *new* frame is available and returns a copy,
    so callers can safely draw on the returned array without corrupting
    the shared buffer.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self.capture = cv2.VideoCapture(self.config.device_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera device {self.config.device_index}")

        # Request MJPG codec — much faster USB transfer than raw YUYV.
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        # Minimize internal frame buffer to avoid stale/laggy frames.
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read one frame synchronously so the first read() never blocks.
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read initial frame from camera")
        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        self._lock = threading.Lock()
        self._latest_frame: np.ndarray = frame
        self._new_frame = threading.Event()
        self._new_frame.set()  # first frame is already available
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._grab_loop, daemon=True)
        self._reader.start()

    def read(self) -> np.ndarray:
        """Block until a new frame is captured, then return a *copy*.

        Blocking here naturally throttles the main loop to the camera's
        native frame rate (~30 fps) and guarantees each iteration gets a
        unique, unmodified frame.
        """
        self._new_frame.wait()
        self._new_frame.clear()
        with self._lock:
            return self._latest_frame.copy()

    def release(self) -> None:
        self._stop.set()
        self._new_frame.set()  # unblock read() if it's waiting
        self._reader.join(timeout=2.0)
        self.capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------

    def _grab_loop(self) -> None:
        """Continuously read frames so USB I/O never blocks the main loop."""
        while not self._stop.is_set():
            ok, frame = self.capture.read()
            if not ok or frame is None:
                continue
            if self.config.flip_horizontal:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._latest_frame = frame
            self._new_frame.set()