"""Training entrypoint for Nationality Detection models.

The functions expect directory datasets compatible with
``tf.keras.utils.image_dataset_from_directory``.
"""

from __future__ import annotations

import argparse
import os

from src.nationality.age_estimator import NationalityAgeEstimator
from src.nationality.dress_colour_classifier import DressColourClassifier
from src.nationality.emotion_predictor import EmotionPredictor
from src.nationality.nationality_detector import NationalityDetector


def _classification_dataset(tf, data_dir: str, image_size, batch_size: int, seed: int):
    return (
        tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.15,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
        ),
        tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.15,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
        ),
    )


def _normalise_dataset(tf, dataset):
    normalise = tf.keras.layers.Rescaling(1.0 / 255.0)
    return dataset.map(lambda x, y: (normalise(x), y))


def train_nationality_detector(data_dir: str, model_dir: str = "models", epochs: int = 30):
    import tensorflow as tf

    train_ds, val_ds = _classification_dataset(tf, data_dir, (128, 128), 32, 42)
    model = NationalityDetector(build_model=True)
    history = model.train(
        _normalise_dataset(tf, train_ds),
        _normalise_dataset(tf, val_ds),
        {"epochs": epochs, "learning_rate": 0.001},
    )
    model.save(os.path.join(model_dir, "nationality_detector.keras"))
    return history


def train_emotion_predictor(data_dir: str, model_dir: str = "models", epochs: int = 50):
    import tensorflow as tf

    train_ds, val_ds = _classification_dataset(tf, data_dir, (48, 48), 64, 42)
    train_ds = train_ds.map(
        lambda x, y: (tf.image.rgb_to_grayscale(tf.cast(x, tf.float32) / 255.0), y)
    )
    val_ds = val_ds.map(
        lambda x, y: (tf.image.rgb_to_grayscale(tf.cast(x, tf.float32) / 255.0), y)
    )
    model = EmotionPredictor(build_model=True)
    history = model.train(train_ds, val_ds, {"epochs": epochs, "learning_rate": 0.001})
    model.save(os.path.join(model_dir, "emotion_predictor.keras"))
    return history


def train_dress_colour_classifier(data_dir: str, model_dir: str = "models", epochs: int = 25):
    import tensorflow as tf

    train_ds, val_ds = _classification_dataset(tf, data_dir, (128, 128), 32, 42)
    model = DressColourClassifier(build_model=True)
    history = model.train(
        _normalise_dataset(tf, train_ds),
        _normalise_dataset(tf, val_ds),
        {"epochs": epochs, "learning_rate": 0.001},
    )
    model.save(os.path.join(model_dir, "dress_colour_classifier.keras"))
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nationality pipeline models.")
    parser.add_argument("--model", choices=["nationality", "emotion", "dress"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    if args.model == "nationality":
        train_nationality_detector(args.data_dir, args.model_dir, args.epochs or 30)
    elif args.model == "emotion":
        train_emotion_predictor(args.data_dir, args.model_dir, args.epochs or 50)
    else:
        train_dress_colour_classifier(args.data_dir, args.model_dir, args.epochs or 25)


if __name__ == "__main__":
    main()
