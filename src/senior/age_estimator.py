"""
Senior Age Estimator - Inference wrapper for age estimation.

Wraps the existing AgeEstimator model (MobileNetV2 backbone) to provide
age prediction with confidence scores for the Senior Citizen Identification system.
"""

import os
import logging
from typing import Tuple, Optional

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except (ImportError, Exception) as e:
    TENSORFLOW_AVAILABLE = False
    _import_error = e
    tf = None
    keras = None

from src.senior.exceptions import ModelPredictionError

logger = logging.getLogger(__name__)


class SeniorAgeEstimator:
    """
    Age estimation model wrapper for the Senior Citizen Identification system.

    Wraps the existing AgeEstimator architecture (MobileNetV2 backbone) and adds
    confidence output derived from the model's prediction certainty.

    Architecture:
    - MobileNetV2 (ImageNet pretrained, no top, 224x224x3 input)
    - GlobalAveragePooling2D
    - Dense(256, activation='relu')
    - Dropout(0.3)
    - Dense(1, activation='relu') for age output

    Returns:
        Tuple[int, float]: (age, confidence) where age is in [1, 100] and
        confidence is in [0.0, 1.0].
    """

    INPUT_SHAPE = (224, 224, 3)

    def __init__(self):
        """Initialize the SeniorAgeEstimator."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                f"TensorFlow is required but not available: {_import_error}"
            )
        self.model: Optional[object] = None
        self._build_model()

    def _build_model(self) -> None:
        """Build the MobileNetV2-based age estimation model."""
        input_tensor = layers.Input(shape=self.INPUT_SHAPE)

        backbone = MobileNetV2(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=False
        )

        x = backbone.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu', name='age_dense_256')(x)
        x = layers.Dropout(0.3, name='age_dropout')(x)
        age_output = layers.Dense(1, activation='relu', name='age_output')(x)

        self.model = keras.Model(
            inputs=input_tensor, outputs=age_output, name='senior_age_estimator'
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
                f"Age estimator model file not found: {weights_path}. "
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
                f"Failed to load age estimator model from {weights_path}. "
                f"The file may be corrupted. Error: {str(e)}. "
                "Please run the training script (train.py) to regenerate model weights."
            )
            logger.error(error_msg)
            raise ModelPredictionError(error_msg)

    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Predict age and confidence from a preprocessed image.

        Args:
            image: Preprocessed image array of shape (224, 224, 3), values in [-1, 1].

        Returns:
            Tuple of (age, confidence) where:
            - age: integer in [1, 100]
            - confidence: float in [0.0, 1.0] representing prediction certainty

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

            # Run forward pass
            raw_output = self.model.predict(batch_image, verbose=0)[0, 0]

            # Clamp age to [1, 100] and convert to int
            age = int(np.clip(raw_output, 1, 100))

            # Compute confidence based on how close the raw output is to valid range
            # Higher confidence when prediction is well within bounds,
            # lower confidence for extreme/boundary values
            confidence = self._compute_confidence(raw_output)

            return age, confidence

        except (ValueError, TypeError) as e:
            raise
        except Exception as e:
            raise ModelPredictionError(
                f"Age estimation inference failed: {str(e)}"
            )

    def _compute_confidence(self, raw_output: float) -> float:
        """
        Compute confidence score from the raw model output.

        The confidence is derived from how close the prediction is to a
        reasonable age range. Predictions well within [1, 100] get higher
        confidence; predictions at the boundaries or outside get lower confidence.

        Args:
            raw_output: Raw scalar output from the model.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        # If the output is far outside the valid range, confidence is low
        # If it's well within range, confidence is high
        # Use a sigmoid-like mapping based on distance from boundaries
        if raw_output <= 0:
            confidence = 0.1
        elif raw_output > 100:
            # The further above 100, the lower the confidence
            overshoot = raw_output - 100
            confidence = max(0.1, 1.0 - overshoot / 100.0)
        else:
            # Within valid range: confidence based on distance from boundaries
            # Midrange values (20-80) get highest confidence
            distance_from_edge = min(raw_output - 1, 100 - raw_output)
            # Normalize: 0 at edge -> low confidence, 50 at center -> high confidence
            confidence = min(1.0, 0.5 + distance_from_edge / 100.0)

        # Clamp to [0.0, 1.0]
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return confidence
