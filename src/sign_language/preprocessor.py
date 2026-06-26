"""Preprocessing for ASL hand-sign images."""

from __future__ import annotations

import cv2
import numpy as np


class SignLanguagePreprocessor:
    """Resize and normalise cropped hand images for the sign CNN."""

    TARGET_SIZE = (64, 64)

    def preprocess(self, hand_image: np.ndarray) -> np.ndarray:
        """Return a ``(64, 64, 3)`` float32 image normalised to ``[0, 1]``."""

        if hand_image is None or not isinstance(hand_image, np.ndarray):
            raise ValueError("hand_image must be a numpy array")
        if hand_image.ndim != 3 or hand_image.shape[2] != 3:
            raise ValueError("hand_image must have shape (H, W, 3)")
        if hand_image.shape[0] <= 0 or hand_image.shape[1] <= 0:
            raise ValueError("hand_image dimensions must be positive")

        resized = cv2.resize(hand_image, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0).clip(0.0, 1.0)
