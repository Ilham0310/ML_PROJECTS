"""
Image preprocessing module for the Long-Hair Gender Identification system.

This module implements image loading, resizing, normalization, and augmentation
for the MobileNetV2-based model pipeline.
"""

import numpy as np
import cv2
import os
from typing import Optional

try:
    import tensorflow as tf
    # Verify this is a real TensorFlow, not a MagicMock from tests
    _has_tensorflow = hasattr(tf, '__version__')
except ImportError:
    _has_tensorflow = False


def _mobilenetv2_preprocess(image: np.ndarray) -> np.ndarray:
    """
    MobileNetV2 preprocess_input equivalent: scale pixel values from [0, 255] to [-1, 1].
    Formula: x / 127.5 - 1.0

    This is the exact same computation performed by
    tf.keras.applications.mobilenet_v2.preprocess_input for mode='tf'.
    """
    image = image.astype(np.float32)
    return image / 127.5 - 1.0


class Preprocessor:
    """
    Handles image preprocessing for the ML pipeline.

    Implements:
    - Image loading and validation
    - Resizing to 224×224 (MobileNetV2 input size)
    - MobileNetV2 preprocessing (scaling to -1..1 range)
    - Training-time data augmentation
    """

    TARGET_SIZE = (224, 224)

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image for model inference.

        Steps:
        1. Load image from file path
        2. Resize to 224×224 pixels
        3. Apply MobileNetV2 preprocessing (scale pixel values to -1..1 range)

        Args:
            image_path: Path to the image file

        Returns:
            np.ndarray: Preprocessed image array of shape (224, 224, 3)
                       with pixel values in range [-1, 1]

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image cannot be loaded or is invalid
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image using OpenCV (BGR format)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to target size (224x224)
        image = cv2.resize(image, self.TARGET_SIZE)

        # Apply MobileNetV2 preprocessing: scale to [-1, 1] range
        # MobileNetV2 expects inputs scaled to [-1, 1], not [0, 1]
        image = _mobilenetv2_preprocess(image)

        return image

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply training-time data augmentation to an image.

        Augmentation techniques applied:
        - Random horizontal flip (50% probability)
        - Random rotation (±10 degrees)
        - Random brightness adjustment (±20% brightness jitter)

        Args:
            image: Input image array of shape (224, 224, 3) with values in [-1, 1]

        Returns:
            np.ndarray: Augmented image with same shape and value range
        """
        if image.shape != (224, 224, 3):
            raise ValueError(f"Expected image shape (224, 224, 3), got {image.shape}")

        # Work on a copy to avoid modifying the original
        augmented = image.copy()

        # Convert from [-1, 1] to [0, 255] for OpenCV operations
        # MobileNetV2 preprocess_input scales to [-1, 1], so we reverse it temporarily
        augmented = (augmented + 1.0) * 127.5
        augmented = augmented.astype(np.uint8)

        # 1. Random horizontal flip (50% probability)
        if np.random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)  # 1 = horizontal flip

        # 2. Random rotation (±10 degrees)
        angle = np.random.uniform(-10, 10)
        if abs(angle) > 0.1:  # Only rotate if angle is significant
            height, width = augmented.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, rotation_matrix, (width, height))

        # 3. Random brightness adjustment (±20% brightness jitter)
        brightness_factor = np.random.uniform(0.8, 1.2)  # ±20% brightness
        augmented = augmented.astype(np.float32)
        augmented = augmented * brightness_factor
        augmented = np.clip(augmented, 0, 255)  # Ensure values stay in valid range
        augmented = augmented.astype(np.uint8)

        # Convert back to float32 and scale back to [-1, 1] range
        augmented = augmented.astype(np.float32)
        augmented = _mobilenetv2_preprocess(augmented)

        return augmented

    def preprocess_batch(self, image_paths: list[str], augment_training: bool = False) -> np.ndarray:
        """
        Preprocess a batch of images.

        Args:
            image_paths: List of paths to image files
            augment_training: Whether to apply data augmentation (for training only)

        Returns:
            np.ndarray: Batch of preprocessed images of shape (batch_size, 224, 224, 3)
        """
        batch = []
        for path in image_paths:
            image = self.preprocess(path)
            if augment_training:
                image = self.augment(image)
            batch.append(image)

        return np.array(batch)

    @staticmethod
    def validate_image_file(image_path: str) -> tuple[bool, Optional[str]]:
        """
        Validate if an image file is readable and in a supported format.

        Args:
            image_path: Path to the image file

        Returns:
            tuple: (is_valid, error_message)
                  is_valid is True if file is valid, False otherwise
                  error_message is None if valid, descriptive string if invalid
        """
        if not os.path.exists(image_path):
            return False, f"File does not exist: {image_path}"

        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        _, ext = os.path.splitext(image_path.lower())
        if ext not in valid_extensions:
            return False, f"Unsupported file format. Supported formats: {', '.join(valid_extensions)}"

        # Try to load the image
        try:
            image = cv2.imread(image_path)
            if image is None or not isinstance(image, np.ndarray):
                return False, "Could not read image file (file may be corrupted)"
            return True, None
        except Exception as e:
            return False, f"Error reading image: {str(e)}"
