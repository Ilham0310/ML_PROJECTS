"""Voice gender classifier."""

from __future__ import annotations

import numpy as np

from .voice_model_utils import VoiceKerasMixin, stable_score, validate_features


class VoiceGenderClassifier(VoiceKerasMixin):
    """Dense neural network wrapper for binary voice gender classification."""

    INPUT_DIM = 28
    MALE_THRESHOLD = 0.965

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
                layers.Dense(1, activation="sigmoid"),
            ],
            name="voice_gender_classifier",
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def predict(self, features: np.ndarray) -> str:
        vector = validate_features(features)
        if self.model is None:
            male_probability = stable_score(vector)
        else:
            male_probability = float(
                self.model.predict(np.expand_dims(vector, axis=0), verbose=0)[0][0]
            )
        return "male" if male_probability >= self.MALE_THRESHOLD else "female"
