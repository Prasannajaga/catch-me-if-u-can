from __future__ import annotations

import threading
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
    """Tracks one hand and returns index-finger position.

    MediaPipe inference runs on a dedicated background thread so the main
    game loop never blocks on detection.  Call ``detect(frame)`` every
    frame — it drops the frame into the worker and returns the latest
    known result immediately.
    """

    def __init__(
        self,
        *,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        smoothing: float = 0.35,
        detection_scale: float = 0.5,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(  # type: ignore
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._smoothing = smoothing
        self._detection_scale = detection_scale

        # Smoothing state (only touched from the worker thread).
        self._last_raw: np.ndarray | None = None

        # Thread-safe communication between main thread and worker.
        self._lock = threading.Lock()
        self._latest_result: HandPoint | None = None
        self._pending_frame: np.ndarray | None = None
        self._has_new_frame = threading.Event()
        self._stop = threading.Event()

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API (called from the main thread)
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> HandPoint | None:
        """Submit *frame_bgr* for detection and return the latest result.

        This call is **non-blocking**.  The returned value may be from a
        previous frame if the worker hasn't finished processing yet.
        """
        with self._lock:
            self._pending_frame = frame_bgr
        self._has_new_frame.set()

        with self._lock:
            return self._latest_result

    def close(self) -> None:
        self._stop.set()
        self._has_new_frame.set()  # unblock the worker if it's waiting
        self._worker.join(timeout=2.0)
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Worker loop — processes frames as they arrive."""
        rgb_buffer: np.ndarray | None = None

        while not self._stop.is_set():
            self._has_new_frame.wait()
            self._has_new_frame.clear()

            if self._stop.is_set():
                break

            # Grab the latest frame (skip stale ones automatically).
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                continue

            result = self._process(frame, rgb_buffer)

            with self._lock:
                self._latest_result = result

    def _process(
        self,
        frame_bgr: np.ndarray,
        rgb_buffer: np.ndarray | None,
    ) -> HandPoint | None:
        """Run MediaPipe on a downscaled copy and return smoothed result."""
        h, w = frame_bgr.shape[:2]
        small_w = int(w * self._detection_scale)
        small_h = int(h * self._detection_scale)
        small = cv2.resize(
            frame_bgr, (small_w, small_h), interpolation=cv2.INTER_LINEAR,
        )

        # Reuse RGB buffer to avoid per-frame allocation.
        if rgb_buffer is None or rgb_buffer.shape[:2] != (small_h, small_w):
            rgb_buffer = np.empty((small_h, small_w, 3), dtype=np.uint8)
        cv2.cvtColor(small, cv2.COLOR_BGR2RGB, dst=rgb_buffer)

        results = self._hands.process(rgb_buffer)

        if not results.multi_hand_landmarks:
            return None

        # MediaPipe landmark 8 = index-finger tip (normalised [0, 1]).
        lm = results.multi_hand_landmarks[0].landmark[8]
        current = np.clip(
            np.array([lm.x, lm.y], dtype=np.float32), 0.0, 1.0,
        )

        if self._last_raw is None:
            smoothed = current
        else:
            smoothed = (
                self._smoothing * self._last_raw
                + (1.0 - self._smoothing) * current
            )

        self._last_raw = smoothed.astype(np.float32)
        return HandPoint(float(smoothed[0]), float(smoothed[1]))