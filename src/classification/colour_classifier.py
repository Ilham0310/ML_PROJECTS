"""TensorFlow/Keras classifier for car colour recognition."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from src.car_colour.models import ColourClassifierConfig
from src.exceptions import ClassificationError, ModelLoadError


def build_colour_classifier_cnn(input_shape=(224, 224, 3), num_classes: int = 8):
    """Build the colour classifier CNN used by the training script."""

    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise ImportError("TensorFlow is required to build the colour CNN.") from exc

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(256, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="car_colour_classifier",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class ColourClassifier:
    """Loads and runs an 8-class colour classifier on car crops."""

    COLOUR_LABELS = [
        "black",
        "blue",
        "green",
        "orange",
        "red",
        "silver",
        "white",
        "yellow",
    ]
    INPUT_SIZE = (224, 224)

    def __init__(
        self,
        weights_path: str = "models/colour_classifier.keras",
        model: Optional[object] = None,
        config: Optional[ColourClassifierConfig] = None,
    ) -> None:
        self.config = config or ColourClassifierConfig(weights_path=weights_path)
        self.colour_labels = list(self.config.colour_labels)
        self.model = model
        if model is not None:
            return
        if not os.path.isfile(weights_path):
            raise ModelLoadError(
                f"Colour classifier model failed to load: weights not found at {weights_path}"
            )
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ModelLoadError(
                "TensorFlow is required to load the colour classifier model."
            ) from exc
        try:
            self.model = keras.models.load_model(weights_path, compile=False)
        except Exception as exc:
            raise ModelLoadError(
                f"Colour classifier model failed to load from {weights_path}: {exc}"
            ) from exc

    def preprocess(self, car_crop: np.ndarray) -> np.ndarray:
        """Resize a crop to 224x224 and normalise it to ``[0, 1]``."""

        if car_crop is None or not isinstance(car_crop, np.ndarray):
            raise ValueError("car_crop must be a numpy array")
        if car_crop.ndim != 3 or car_crop.shape[2] != 3:
            raise ValueError("car_crop must have shape (H, W, 3)")
        if car_crop.shape[0] <= 0 or car_crop.shape[1] <= 0:
            raise ValueError("car_crop dimensions must be positive")
        resized = cv2.resize(car_crop, self.INPUT_SIZE, interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0).clip(0.0, 1.0)

    def classify(self, car_crop: np.ndarray) -> str:
        """Return exactly one colour label from ``COLOUR_LABELS``."""

        if self.model is None:
            raise ClassificationError("Colour classifier model is not loaded.")
        try:
            processed = self.preprocess(car_crop)
            raw = self.model.predict(np.expand_dims(processed, axis=0), verbose=0)
            probabilities = np.asarray(raw, dtype=float).reshape(-1)
            if probabilities.size != len(self.colour_labels):
                raise ValueError(
                    "Model output size does not match the colour label set."
                )
            index = int(np.argmax(probabilities))
            return self.colour_labels[index]
        except ClassificationError:
            raise
        except Exception as exc:
            raise ClassificationError(f"Colour classification failed: {exc}") from exc
