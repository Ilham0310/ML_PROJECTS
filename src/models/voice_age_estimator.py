"""Voice age estimator."""

from __future__ import annotations

import numpy as np

from .voice_model_utils import VoiceKerasMixin, validate_features


class VoiceAgeEstimator(VoiceKerasMixin):
    """Dense regression wrapper for speaker age estimation."""

    INPUT_DIM = 28
    AGE_MIN = 10
    AGE_MAX = 100

    def __init__(self, model=None, build_model: bool = False) -> None:
        self.model = model
        if build_model:
            self.model = self._build_model()

    def _build_model(self):
        keras = self._require_keras()
        layers = keras.layers
        model = keras.Sequential(
            [
                layers.Input(shape=(self.INPUT_DIM,)),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(32, activation="relu"),
                layers.Dense(1, activation="linear"),
            ],
            name="voice_age_estimator",
        )
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        return model

    def predict(self, features: np.ndarray) -> int:
        vector = validate_features(features)
        if self.model is None:
            raw_age = 55.0 + float(np.tanh(np.mean(vector))) * 20.0
        else:
            raw_age = float(
                self.model.predict(np.expand_dims(vector, axis=0), verbose=0)[0][0]
            )
        return int(np.clip(raw_age, self.AGE_MIN, self.AGE_MAX))
