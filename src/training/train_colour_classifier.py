"""Train the car colour classifier from cropped car images."""

from __future__ import annotations

import argparse
import os

from src.classification.colour_classifier import ColourClassifier, build_colour_classifier_cnn


def train_colour_classifier(
    data_dir: str = "data/crop_part1",
    output_path: str = "models/colour_classifier.keras",
    epochs: int = 25,
    batch_size: int = 32,
    seed: int = 42,
) -> None:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for colour classifier training.") from exc

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=ColourClassifier.INPUT_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=ColourClassifier.INPUT_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
    )
    normalise = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (normalise(x), y))
    val_ds = val_ds.map(lambda x, y: (normalise(x), y))

    model = build_colour_classifier_cnn(num_classes=len(ColourClassifier.COLOUR_LABELS))
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    _, val_accuracy = model.evaluate(val_ds, verbose=0)
    if val_accuracy < 0.70:
        raise RuntimeError(
            f"Validation accuracy {val_accuracy:.3f} is below required 0.700."
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the car colour classifier.")
    parser.add_argument("--data-dir", default="data/crop_part1")
    parser.add_argument("--output-path", default="models/colour_classifier.keras")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_colour_classifier(**vars(args))


if __name__ == "__main__":
    main()
