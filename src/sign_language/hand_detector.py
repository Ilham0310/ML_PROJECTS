"""MediaPipe-based hand detection and cropping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False


@dataclass(frozen=True)
class HandBoundingBox:
    """Pixel-space hand bounding box."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def clipped(self, image_width: int, image_height: int) -> "HandBoundingBox":
        """Return this box clipped to image bounds."""

        x1 = min(max(0, self.x), image_width)
        y1 = min(max(0, self.y), image_height)
        x2 = min(max(0, self.x + self.width), image_width)
        y2 = min(max(0, self.y + self.height), image_height)
        return HandBoundingBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


class HandDetector:
    """Detects the largest hand in a BGR image using MediaPipe Hands."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        padding: int = 20,
    ) -> None:
        if max_num_hands < 1:
            raise ValueError("max_num_hands must be at least 1")
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be in [0, 1]")
        if padding < 0:
            raise ValueError("padding must be non-negative")

        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.padding = padding
        self._hands = None

        if MEDIAPIPE_AVAILABLE:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
            )

    @staticmethod
    def select_largest_bbox(
        boxes: Iterable[HandBoundingBox],
    ) -> Optional[HandBoundingBox]:
        """Select the bounding box with largest area, or ``None`` if empty."""

        return max(list(boxes), key=lambda box: box.area, default=None)

    @staticmethod
    def _landmarks_to_bbox(
        landmarks: Sequence,
        image_shape: tuple[int, int, int],
        padding: int = 20,
    ) -> HandBoundingBox:
        height, width = image_shape[:2]
        xs = [landmark.x for landmark in landmarks]
        ys = [landmark.y for landmark in landmarks]
        x1 = int(min(xs) * width) - padding
        y1 = int(min(ys) * height) - padding
        x2 = int(max(xs) * width) + padding
        y2 = int(max(ys) * height) + padding
        return HandBoundingBox(x1, y1, x2 - x1, y2 - y1).clipped(width, height)

    def detect_bbox(self, frame: np.ndarray) -> Optional[HandBoundingBox]:
        """Return the largest detected hand box for a BGR frame."""

        if self._hands is None:
            raise ImportError(
                "MediaPipe is required for hand detection. "
                "Install it with `pip install mediapipe`."
            )
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3)")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None

        boxes = [
            self._landmarks_to_bbox(
                hand_landmarks.landmark,
                frame.shape,
                padding=self.padding,
            )
            for hand_landmarks in results.multi_hand_landmarks
        ]
        return self.select_largest_bbox(boxes)

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return the cropped largest hand region, or ``None`` when absent."""

        bbox = self.detect_bbox(frame)
        if bbox is None or bbox.area == 0:
            return None
        return frame[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width].copy()

    def close(self) -> None:
        """Release MediaPipe resources."""

        if self._hands is not None:
            self._hands.close()
            self._hands = None
