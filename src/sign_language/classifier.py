"""Classifier wrapper and display helpers for ASL sign detection."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tensorflow import keras

    TENSORFLOW_AVAILABLE = True
except ImportError:
    keras = None
    TENSORFLOW_AVAILABLE = False


LOW_CONFIDENCE_MESSAGE = "Low confidence - sign not recognized"
DEFAULT_ASL_LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")


class SignLanguageModelLoadError(Exception):
    """Raised when sign-language model weights cannot be loaded."""


def normalise_probabilities(values: Sequence[float]) -> np.ndarray:
    """Convert arbitrary model outputs into a stable probability distribution."""

    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    if raw.size == 0:
        raise ValueError("probability vector must not be empty")
    if np.any(~np.isfinite(raw)):
        raise ValueError("probability vector contains non-finite values")

    if np.all(raw >= 0.0) and raw.sum() > 0.0:
        probs = raw / raw.sum()
    else:
        shifted = raw - raw.max()
        exp_values = np.exp(shifted)
        probs = exp_values / exp_values.sum()
    return probs.astype(np.float32)


def format_prediction(
    label: str,
    confidence: float,
    threshold: float = 0.50,
) -> str:
    """Format a prediction for display, applying the low-confidence rule."""

    if confidence < threshold:
        return LOW_CONFIDENCE_MESSAGE
    return f"{label} - {confidence * 100:.0f}%"


class SignLanguageClassifier:
    """Keras CNN wrapper for ASL alphabet classification."""

    CONFIDENCE_THRESHOLD = 0.50

    def __init__(
        self,
        model_path: Optional[str] = None,
        class_labels: Optional[Iterable[str]] = None,
        model=None,
    ) -> None:
        self.class_labels: List[str] = list(class_labels or DEFAULT_ASL_LABELS)
        if len(self.class_labels) < 1:
            raise ValueError("class_labels must not be empty")

        self.model = model
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Load a saved Keras model from disk."""

        if not os.path.isfile(model_path):
            raise SignLanguageModelLoadError(
                f"Model not found: {model_path}. Please run training first."
            )
        if not TENSORFLOW_AVAILABLE:
            raise SignLanguageModelLoadError(
                "TensorFlow is required to load sign-language model weights."
            )
        try:
            self.model = keras.models.load_model(model_path, compile=False)
        except Exception as exc:
            raise SignLanguageModelLoadError(
                f"Failed to load sign-language model from {model_path}: {exc}"
            ) from exc

    def predict_probabilities(self, processed_image: np.ndarray) -> np.ndarray:
        """Run inference and return a probability vector over class labels."""

        if self.model is None:
            raise SignLanguageModelLoadError(
                "Model not found. Please run training first."
            )
        if processed_image.shape != (64, 64, 3):
            raise ValueError(
                f"Expected processed_image shape (64, 64, 3), got {processed_image.shape}"
            )

        batch = np.expand_dims(processed_image.astype(np.float32), axis=0)
        raw = self.model.predict(batch, verbose=0)
        probs = normalise_probabilities(np.asarray(raw)[0])
        if probs.size != len(self.class_labels):
            raise ValueError(
                "Model output size does not match number of class labels: "
                f"{probs.size} != {len(self.class_labels)}"
            )
        return probs

    def predict(self, processed_image: np.ndarray) -> Tuple[str, float]:
        """Return ``(label, confidence)`` after applying confidence thresholding."""

        probs = self.predict_probabilities(processed_image)
        index = int(np.argmax(probs))
        confidence = float(probs[index])
        label = self.class_labels[index]
        if confidence < self.CONFIDENCE_THRESHOLD:
            return LOW_CONFIDENCE_MESSAGE, confidence
        return label, confidence

    @staticmethod
    def display_label(label: str, confidence: float) -> str:
        """Return the GUI display string for a classifier output."""

        return format_prediction(
            label,
            confidence,
            threshold=SignLanguageClassifier.CONFIDENCE_THRESHOLD,
        )
