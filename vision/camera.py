from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    flip_horizontal: bool = True


class Camera:
    """Small wrapper around cv2.VideoCapture."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self.capture = cv2.VideoCapture(self.config.device_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera device {self.config.device_index}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

    def read(self) -> np.ndarray:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera")

        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()