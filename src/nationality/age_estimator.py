"""Age estimator for the nationality routing pipeline."""

from __future__ import annotations

import numpy as np

from .model_utils import KerasModelMixin, validate_tensor


class NationalityAgeEstimator(KerasModelMixin):
    """CNN regression wrapper returning age in [1, 120]."""

    INPUT_SHAPE = (128, 128, 3)
    _loss_name = "mae"
    _metrics = ["mae"]

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
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(128, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1, activation="relu"),
            ],
            name="nationality_age_estimator",
        )
        return model

    def predict(self, image: np.ndarray) -> int:
        image = validate_tensor(image, self.INPUT_SHAPE, "image")
        if self.model is None:
            raw_age = 1 + float(np.mean(image)) * 119
        else:
            raw_age = float(self.model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0])
        return int(np.clip(raw_age, 1, 120))
