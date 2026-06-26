"""Image preprocessing utilities for Nationality Detection."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class NationalityPreprocessor:
    """Preprocesses face and full-image inputs for the nationality pipeline."""

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("image dimensions must be positive")

    def preprocess_face(
        self,
        face_roi: np.ndarray,
        target_size: Tuple[int, int] = (128, 128),
    ) -> np.ndarray:
        """Resize BGR face ROI, convert to RGB, and normalise to [0, 1]."""

        self._validate_image(face_roi)
        resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)

    def preprocess_face_grayscale(
        self,
        face_roi: np.ndarray,
        target_size: Tuple[int, int] = (48, 48),
    ) -> np.ndarray:
        """Resize BGR face ROI to grayscale and normalise to [0, 1]."""

        self._validate_image(face_roi)
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        normalised = (resized.astype(np.float32) / 255.0).clip(0.0, 1.0)
        return np.expand_dims(normalised, axis=-1)

    def preprocess_full_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (128, 128),
    ) -> np.ndarray:
        """Resize full BGR image, convert to RGB, and normalise to [0, 1]."""

        return self.preprocess_face(image, target_size=target_size)
