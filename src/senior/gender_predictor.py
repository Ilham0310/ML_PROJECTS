"""
Senior Gender Predictor - Inference wrapper for gender classification.

Wraps the existing GenderPredictor model (MobileNetV2 backbone) to provide
gender prediction with confidence scores and "Unknown" handling for the
Senior Citizen Identification system.
"""

import os
import logging
from typing import Tuple, Optional

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    TENSORFLOW_AVAILABLE = True
except (ImportError, Exception) as e:
    TENSORFLOW_AVAILABLE = False
    _import_error = e
    tf = None
    keras = None

from src.senior.exceptions import ModelPredictionError

logger = logging.getLogger(__name__)


class SeniorGenderPredictor:
    """
    Gender classification model wrapper for the Senior Citizen Identification system.

    Wraps the existing GenderPredictor architecture (MobileNetV2 backbone) and adds
    "Unknown" gender handling when confidence is below 0.4.

    Architecture:
    - MobileNetV2 (ImageNet pretrained, no top, 224x224x3 input)
    - GlobalAveragePooling2D
    - Dense(128, activation='relu')
    - Dropout(0.3)
    - Dense(1, activation='sigmoid')

    Sigmoid output mapping:
    - output < 0.5 -> "Female"
    - output >= 0.5 -> "Male"
    - Confidence = max(output, 1 - output)
    - If confidence < 0.4 -> returns "Unknown"

    Returns:
        Tuple[str, float]: (gender_label, confidence) where gender_label is
        "Male", "Female", or "Unknown", and confidence is in [0.0, 1.0].
    """

    INPUT_SHAPE = (224, 224, 3)
    UNKNOWN_CONFIDENCE_THRESHOLD = 0.4

    def __init__(self):
        """Initialize the SeniorGenderPredictor."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                f"TensorFlow is required but not available: {_import_error}"
            )
        self.model: Optional[object] = None
        self._build_model()

    def _build_model(self) -> None:
        """Build the MobileNetV2-based gender classification model."""
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=self.INPUT_SHAPE
        )

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', name='gender_dense_1')(x)
        x = Dropout(0.3, name='gender_dropout')(x)
        predictions = Dense(1, activation='sigmoid', name='gender_output')(x)

        self.model = Model(
            inputs=base_model.input, outputs=predictions, name='senior_gender_predictor'
        )

    def load(self, weights_path: str) -> None:
        """
        Load model weights from file.

        Args:
            weights_path: Path to the model weights file (.keras or .h5)

        Raises:
            ModelPredictionError: If the file is missing or corrupted.
        """
        if not os.path.exists(weights_path):
            error_msg = (
                f"Gender predictor model file not found: {weights_path}. "
                "Please run the training script (train.py) to generate model weights."
            )
            logger.error(error_msg)
            raise ModelPredictionError(error_msg)

        try:
            self.model = keras.models.load_model(
                weights_path,
                compile=False
            )
        except Exception as e:
            error_msg = (
                f"Failed to load gender predictor model from {weights_path}. "
                f"The file may be corrupted. Error: {str(e)}. "
                "Please run the training script (train.py) to regenerate model weights."
            )
            logger.error(error_msg)
            raise ModelPredictionError(error_msg)

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict gender and confidence from a preprocessed image.

        Args:
            image: Preprocessed image array of shape (224, 224, 3), values in [-1, 1].

        Returns:
            Tuple of (gender_label, confidence) where:
            - gender_label: "Male", "Female", or "Unknown" (if confidence < 0.4)
            - confidence: float in [0.0, 1.0]

        Raises:
            ValueError: If model is not loaded or image has wrong shape.
            ModelPredictionError: If inference fails.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        if image.shape != self.INPUT_SHAPE:
            raise ValueError(
                f"Expected image shape {self.INPUT_SHAPE}, got {image.shape}"
            )

        try:
            # Add batch dimension
            batch_image = np.expand_dims(image, axis=0)

            # Run forward pass - sigmoid output
            prediction = self.model.predict(batch_image, verbose=0)[0][0]

            # Determine gender label and confidence from sigmoid output
            # output < 0.5 -> "Female", output >= 0.5 -> "Male"
            # Confidence = max(output, 1 - output)
            if prediction < 0.5:
                raw_label = "Female"
                confidence = float(1.0 - prediction)
            else:
                raw_label = "Male"
                confidence = float(prediction)

            # Clamp confidence to [0.0, 1.0]
            confidence = float(np.clip(confidence, 0.0, 1.0))

            # Apply Unknown threshold
            if confidence < self.UNKNOWN_CONFIDENCE_THRESHOLD:
                gender_label = "Unknown"
            else:
                gender_label = raw_label

            return gender_label, confidence

        except (ValueError, TypeError) as e:
            raise
        except Exception as e:
            raise ModelPredictionError(
                f"Gender prediction inference failed: {str(e)}"
            )
