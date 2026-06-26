"""Shared model helpers for Nationality Detection."""

from __future__ import annotations

import os
from typing import Iterable, Sequence, Tuple

import numpy as np

from .exceptions import ModelLoadError, MODEL_LOAD_MESSAGE


def stable_probabilities(image: np.ndarray, num_classes: int) -> np.ndarray:
    """Deterministic fallback probabilities used when TensorFlow is unavailable."""

    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    mean = float(np.mean(image)) if image.size else 0.0
    base = np.linspace(0.2, 1.0, num_classes, dtype=np.float64)
    rotated = np.roll(base, int(mean * 1000) % num_classes)
    probs = rotated / rotated.sum()
    return probs.astype(np.float32)


def validate_tensor(
    image: np.ndarray,
    expected_shape: Tuple[int, int, int],
    name: str,
) -> np.ndarray:
    """Validate model input shape and normalise dtype."""

    if image is None or not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if image.shape != expected_shape:
        raise ValueError(f"Expected {name} shape {expected_shape}, got {image.shape}")
    image = image.astype(np.float32)
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError(f"{name} values must be in [0, 1]")
    return image


class KerasModelMixin:
    """Small mixin for optional Keras model loading/saving/training."""

    model = None

    def _require_tensorflow(self):
        try:
            from tensorflow import keras

            return keras
        except ImportError as exc:
            raise ModelLoadError(MODEL_LOAD_MESSAGE) from exc

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise ModelLoadError(MODEL_LOAD_MESSAGE)
        keras = self._require_tensorflow()
        try:
            self.model = keras.models.load_model(path, compile=False)
        except Exception as exc:
            raise ModelLoadError(MODEL_LOAD_MESSAGE) from exc

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not built or loaded.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def train(self, train_ds, val_ds, config: dict):
        if self.model is None:
            self.model = self._build_model()
        learning_rate = config.get("learning_rate", 0.001)
        epochs = config.get("epochs", 1)
        keras = self._require_tensorflow()
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self._loss_name,
            metrics=self._metrics,
        )
        return self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    @staticmethod
    def _predict_label_from_probs(
        probabilities: Sequence[float],
        classes: Sequence[str],
    ) -> tuple[str, float]:
        probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if probs.size != len(classes):
            raise ValueError("Model output size does not match class labels.")
        if probs.sum() <= 0:
            probs = np.ones_like(probs) / probs.size
        else:
            probs = probs / probs.sum()
        index = int(np.argmax(probs))
        return classes[index], float(probs[index])
