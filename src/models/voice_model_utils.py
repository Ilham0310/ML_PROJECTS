"""Shared helpers for voice model wrappers."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np


class VoiceModelLoadError(Exception):
    """Raised when voice model files are missing or corrupted."""


def validate_features(features: np.ndarray) -> np.ndarray:
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    if vector.shape != (28,):
        raise ValueError(f"Expected feature vector shape (28,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Feature vector contains non-finite values")
    return vector


def stable_score(features: np.ndarray) -> float:
    vector = validate_features(features)
    return float(1.0 / (1.0 + np.exp(-np.mean(vector))))


class VoiceKerasMixin:
    """Optional Keras load/save support with NumPy fallback prediction."""

    model = None

    def _require_keras(self):
        try:
            from tensorflow import keras

            return keras
        except ImportError as exc:
            raise VoiceModelLoadError("Model files not found. Please train models first.") from exc

    def load(self, weights_path: str) -> None:
        if not os.path.isfile(weights_path):
            raise VoiceModelLoadError("Model files not found. Please train models first.")
        keras = self._require_keras()
        try:
            self.model = keras.models.load_model(weights_path, compile=False)
        except Exception as exc:
            raise VoiceModelLoadError("Model files corrupted. Please retrain models.") from exc

    def save(self, weights_path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not built or loaded.")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        self.model.save(weights_path)

    @staticmethod
    def _label_from_probabilities(probabilities: Sequence[float], labels: Sequence[str]) -> str:
        probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if probs.size != len(labels):
            raise ValueError("Model output size does not match labels")
        return labels[int(np.argmax(probs))]
