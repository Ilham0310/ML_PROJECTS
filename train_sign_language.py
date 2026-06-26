"""Train the Sign Language Detection CNN.

Expected dataset layout:

data/asl_alphabet/
  A/
  B/
  ...
"""

from __future__ import annotations

import argparse
import json
import os

from src.sign_language.model import build_sign_language_cnn


def train(
    data_dir: str = "data/asl_alphabet",
    model_path: str = "models/sign_language_cnn.keras",
    config_path: str = "models/sign_language_config.json",
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 42,
) -> None:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to train the model.") from exc

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_labels = sorted(
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    )
    if len(class_labels) < 10:
        raise ValueError("ASL dataset must contain at least 10 class directories.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode="categorical",
    )

    normalise = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (normalise(x), y))
    val_ds = val_ds.map(lambda x, y: (normalise(x), y))

    model = build_sign_language_cnn(num_classes=len(class_labels))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    _, val_accuracy = model.evaluate(val_ds, verbose=0)
    if val_accuracy < 0.80:
        raise RuntimeError(
            f"Validation accuracy {val_accuracy:.3f} is below required 0.800."
        )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    config = {
        "num_classes": len(class_labels),
        "class_labels": class_labels,
        "input_shape": [64, 64, 3],
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
        },
        "accuracy_achieved": float(val_accuracy),
        "seed": seed,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sign-language CNN.")
    parser.add_argument("--data-dir", default="data/asl_alphabet")
    parser.add_argument("--model-path", default="models/sign_language_cnn.keras")
    parser.add_argument("--config-path", default="models/sign_language_config.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
