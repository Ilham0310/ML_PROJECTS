"""Training script for voice gender, age, and emotion models."""

from __future__ import annotations

import argparse
import os

from src.models.voice_age_estimator import VoiceAgeEstimator
from src.models.voice_emotion_detector import VoiceEmotionDetector
from src.models.voice_gender_classifier import VoiceGenderClassifier


def _load_feature_csv(path: str):
    import pandas as pd

    frame = pd.read_csv(path)
    feature_cols = [col for col in frame.columns if col.startswith("f")]
    if len(feature_cols) != 28:
        raise ValueError("Training CSV must contain feature columns f0 through f27.")
    return frame, feature_cols


def _train_classifier(model_wrapper, csv_path: str, label_col: str, output_path: str, epochs: int):
    import numpy as np
    from tensorflow import keras

    frame, feature_cols = _load_feature_csv(csv_path)
    x = frame[feature_cols].to_numpy(dtype="float32")
    labels = frame[label_col].to_numpy()

    if label_col == "gender":
        y = (labels == "male").astype("float32")
    else:
        classes = model_wrapper.EMOTIONS
        y = keras.utils.to_categorical([classes.index(label) for label in labels], len(classes))

    model_wrapper.model = model_wrapper._build_model()
    model_wrapper.model.fit(x, y, epochs=epochs, validation_split=0.2)
    model_wrapper.save(output_path)


def _train_age(csv_path: str, output_path: str, epochs: int):
    frame, feature_cols = _load_feature_csv(csv_path)
    x = frame[feature_cols].to_numpy(dtype="float32")
    y = frame["age"].to_numpy(dtype="float32")
    model = VoiceAgeEstimator(build_model=True)
    model.model.fit(x, y, epochs=epochs, validation_split=0.2)
    model.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train voice analysis models.")
    parser.add_argument("--csv", required=True, help="CSV with f0..f27 and labels")
    parser.add_argument("--model", choices=["gender", "age", "emotion"], required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    if args.model == "gender":
        _train_classifier(
            VoiceGenderClassifier(),
            args.csv,
            "gender",
            os.path.join(args.models_dir, "voice_gender_classifier.keras"),
            args.epochs,
        )
    elif args.model == "age":
        _train_age(
            args.csv,
            os.path.join(args.models_dir, "voice_age_estimator.keras"),
            args.epochs,
        )
    else:
        _train_classifier(
            VoiceEmotionDetector(),
            args.csv,
            "emotion",
            os.path.join(args.models_dir, "voice_emotion_detector.keras"),
            args.epochs,
        )


if __name__ == "__main__":
    main()
