"""Emotion Predictor CNN wrapper."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .model_utils import KerasModelMixin, stable_probabilities, validate_tensor


class EmotionPredictor(KerasModelMixin):
    """Custom CNN classifier for facial emotion."""

    INPUT_SHAPE = (48, 48, 1)
    CLASSES = ["happy", "sad", "angry", "surprised", "neutral", "fearful", "disgusted"]
    _loss_name = "categorical_crossentropy"
    _metrics = ["accuracy"]

    def __init__(self, model=None, build_model: bool = False) -> None:
        self.model = model
        if build_model:
            self.model = self._build_model()

    def _build_model(self):
        keras = self._require_tensorflow()
        layers = keras.layers
        model = keras.Sequential(
            [
                layers.Input(shape=self.INPUT_SHAPE),
                layers.Conv2D(32, 3, activation="relu"),
                layers.Conv2D(64, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.Dropout(0.25),
                layers.Conv2D(128, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(len(self.CLASSES), activation="softmax"),
            ],
            name="emotion_predictor",
        )
        return model

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        image = validate_tensor(image, self.INPUT_SHAPE, "image")
        if self.model is None:
            probabilities = stable_probabilities(image, len(self.CLASSES))
        else:
            probabilities = self.model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        return self._predict_label_from_probs(probabilities, self.CLASSES)
