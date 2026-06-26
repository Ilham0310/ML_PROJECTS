"""CNN architecture for Sign Language Detection."""

from __future__ import annotations

from typing import Tuple


def build_sign_language_cnn(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (64, 64, 3),
):
    """Build and compile the custom ASL sign CNN."""

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build the sign-language CNN."
        ) from exc

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="sign_language_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
