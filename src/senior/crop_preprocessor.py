"""Crop and preprocess detected person regions for model inference.

Extracts person ROI from frames using bounding box coordinates,
resizes to 224x224 pixels, and normalizes to [-1, 1] for MobileNetV2 input.
"""

import numpy as np
import cv2

from src.senior.models import BoundingBox


class CropPreprocessor:
    """Preprocesses cropped person regions for MobileNetV2-based models.

    Extracts the person ROI from a video frame using bounding box coordinates,
    resizes the crop to 224x224 pixels, and normalizes pixel values to the
    [-1, 1] range expected by MobileNetV2.
    """

    TARGET_SIZE = (224, 224)

    def crop_and_preprocess(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Crop person from frame, resize to 224x224, normalize to [-1, 1].

        Args:
            frame: Input video frame as a numpy array (H, W, 3) in BGR format.
            bbox: BoundingBox with x, y, width, height coordinates.

        Returns:
            float32 ndarray of shape (224, 224, 3) with values in [-1, 1].

        Raises:
            ValueError: If the bounding box results in a zero-area crop after
                boundary clipping.
        """
        frame_h, frame_w = frame.shape[:2]

        # Clip bounding box to frame boundaries
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(frame_w, bbox.x + bbox.width)
        y2 = min(frame_h, bbox.y + bbox.height)

        # Validate that the clipped region has non-zero area
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Bounding box results in zero-area crop after clipping. "
                f"Clipped region: x1={x1}, y1={y1}, x2={x2}, y2={y2}"
            )

        # Extract the person ROI
        crop = frame[y1:y2, x1:x2]

        # Resize to 224x224
        resized = cv2.resize(crop, self.TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        # Normalize to [-1, 1] using MobileNetV2 preprocessing: x / 127.5 - 1.0
        normalized = resized.astype(np.float32) / 127.5 - 1.0

        return normalized
