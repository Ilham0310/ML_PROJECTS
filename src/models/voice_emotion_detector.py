"""Voice emotion detector."""

from __future__ import annotations

import numpy as np

from .voice_model_utils import VoiceKerasMixin, stable_score, validate_features


class VoiceEmotionDetector(VoiceKerasMixin):
    """Dense classifier wrapper for seven voice emotion labels."""

    INPUT_DIM = 28
    EMOTIONS = ["happy", "sad", "angry", "fearful", "neutral", "surprised", "disgusted"]

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
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.4),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(len(self.EMOTIONS), activation="softmax"),
            ],
            name="voice_emotion_detector",
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def predict(self, features: np.ndarray) -> str:
        vector = validate_features(features)
        if self.model is None:
            base = np.linspace(0.2, 1.0, len(self.EMOTIONS), dtype=np.float32)
            probabilities = np.roll(base, int(stable_score(vector) * 100) % len(self.EMOTIONS))
        else:
            probabilities = self.model.predict(np.expand_dims(vector, axis=0), verbose=0)[0]
        return self._label_from_probabilities(probabilities, self.EMOTIONS)
