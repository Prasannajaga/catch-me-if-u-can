from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

@dataclass
class HandPoint:
    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32) 

class HandTracker:
    """Tracks one hand and returns index-finger position."""

    def __init__(
        self,
        *,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        smoothing: float = 0.75,
    ) -> None:
        self.mp = mp
        self.hands = mp.solutions.hands.Hands(  # type: ignore
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.smoothing = smoothing
        self._last_point: np.ndarray | None = None

    def detect(self, frame_bgr: np.ndarray) -> HandPoint | None:
        """Return normalized index-finger tip position, or None if no hand is found."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        # MediaPipe Hands landmark 8 is the index-finger tip.
        landmark = results.multi_hand_landmarks[0].landmark[8]
        current = np.array([landmark.x, landmark.y], dtype=np.float32)
        current = np.clip(current, 0.0, 1.0)

        if self._last_point is None:
            smoothed = current
        else:
            smoothed = self.smoothing * self._last_point + (1.0 - self.smoothing) * current

        self._last_point = smoothed.astype(np.float32)
        return HandPoint(float(smoothed[0]), float(smoothed[1]))

    def close(self) -> None:
        self.hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()