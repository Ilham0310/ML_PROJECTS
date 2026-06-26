#!/usr/bin/env python3
"""
Senior Citizen Identification - Model Training Pipeline

Trains two MobileNetV2-based models for the senior citizen identification system:
- Age Estimator: regression model predicting age in [1, 100] (Huber loss)
- Gender Classifier: binary classifier predicting Male/Female (binary crossentropy)

Dataset: UTKFace (format: {age}_{gender}_{race}_{timestamp}.jpg.chip.jpg)
Split: 70% train, 15% validation, 15% test (stratified by age group and gender)
Output: models/senior_age_estimator.keras, models/senior_gender_predictor.keras

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8
"""

import sys
import os
import json
import argparse
import random
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except (ImportError, Exception) as e:
    TENSORFLOW_AVAILABLE = False
    _tf_import_error = e

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = "data"
DEFAULT_MODEL_DIR = "models"
DEFAULT_AGE_MODEL_NAME = "senior_age_estimator.keras"
DEFAULT_GENDER_MODEL_NAME = "senior_gender_predictor.keras"

# Split ratios (Requirement 7.3)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility (Requirement 7.6)."""
    random.seed(seed)
    np.random.seed(seed)
    if TENSORFLOW_AVAILABLE:
        tf.random.set_seed(seed)


def _mobilenetv2_preprocess(image: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [-1, 1] for MobileNetV2."""
    image = image.astype(np.float32)
    return image / 127.5 - 1.0


class SeniorDatasetLoader:
    """
    Loads and processes UTKFace dataset for the senior citizen identification task.

    Parses UTKFace filenames ({age}_{gender}_{race}_{timestamp}.jpg.chip.jpg),
    extracts age and gender labels, and performs stratified splitting.
    """

    def __init__(self, data_dir: str, seed: int):
        """
        Initialize the dataset loader.

        Args:
            data_dir: Base directory containing UTKFace data
            seed: Random seed for reproducible splitting
        """
        self.data_dir = data_dir
        self.seed = seed

    def _find_dataset_dir(self) -> str:
        """Locate the UTKFace image directory."""
        # Check common directory names
        candidates = [
            os.path.join(self.data_dir, "UTKFace"),
            os.path.join(self.data_dir, "utkface_aligned_cropped", "UTKFace"),
            os.path.join(self.data_dir, "crop_part1"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate) and os.listdir(candidate):
                return candidate
        raise FileNotFoundError(
            f"UTKFace dataset not found. Searched: {candidates}. "
            "Please download and extract the UTKFace dataset to the data/ directory."
        )

    def _parse_filename(self, filename: str) -> Optional[Tuple[int, int]]:
        """
        Parse UTKFace filename to extract age and gender.

        Format: {age}_{gender}_{race}_{timestamp}.jpg.chip.jpg
        Gender: 0 = Male, 1 = Female

        Args:
            filename: Image filename

        Returns:
            Tuple of (age, gender_int) or None if malformed
        """
        try:
            # Remove extensions
            base_name = filename.replace(".jpg.chip.jpg", "")
            if base_name.endswith(".jpg"):
                base_name = base_name[:-4]

            parts = base_name.split("_")
            if len(parts) < 2:
                return None

            age = int(parts[0])
            gender = int(parts[1])

            # Validate ranges
            if age < 1:
                age = 1
            if age > 100:
                age = 100
            if gender not in (0, 1):
                return None

            return age, gender
        except (ValueError, IndexError):
            return None

    def load(self) -> pd.DataFrame:
        """
        Load the UTKFace dataset, parsing age and gender from filenames.

        Returns:
            DataFrame with columns: path, age, gender
        """
        dataset_dir = self._find_dataset_dir()
        logger.info(f"Loading dataset from: {dataset_dir}")

        image_files = [
            f for f in os.listdir(dataset_dir)
            if f.endswith(".jpg.chip.jpg") or f.endswith(".jpg")
        ]

        if not image_files:
            raise ValueError(f"No image files found in {dataset_dir}")

        data = []
        malformed_count = 0

        for filename in image_files:
            parsed = self._parse_filename(filename)
            if parsed is None:
                malformed_count += 1
                continue

            age, gender_int = parsed
            image_path = os.path.join(dataset_dir, filename)
            gender_label = "Female" if gender_int == 1 else "Male"

            data.append({
                "path": image_path,
                "age": age,
                "gender": gender_label,
            })

        if malformed_count > 0:
            logger.warning(f"Skipped {malformed_count} malformed filenames")

        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} samples (age range: {df['age'].min()}-{df['age'].max()})")

        if len(df) == 0:
            raise ValueError("No valid samples found after parsing filenames")

        return df

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified split: 70% train, 15% validation, 15% test.

        Stratification is by age group and gender (Requirement 7.3).
        Age groups: [1-20], [21-40], [41-60], [61-80], [81-100]

        Args:
            df: DataFrame with columns: path, age, gender

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = df.copy()

        # Create age group bins for stratification
        bins = [0, 20, 40, 60, 80, 101]
        labels = ["0-20", "21-40", "41-60", "61-80", "81-100"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
        df["strat_key"] = df["age_group"].astype(str) + "_" + df["gender"]

        # Filter strata with too few samples (need at least 3 for splitting)
        strat_counts = df["strat_key"].value_counts()
        insufficient = strat_counts[strat_counts < 3]
        if len(insufficient) > 0:
            logger.warning(
                f"Removing {len(insufficient)} strata with <3 samples: "
                f"{insufficient.to_dict()}"
            )
            df = df[~df["strat_key"].isin(insufficient.index)]

        if len(df) == 0:
            raise ValueError("No samples remaining after filtering insufficient strata")

        # First split: separate test set (15%)
        train_val_df, test_df = train_test_split(
            df,
            test_size=TEST_RATIO,
            random_state=self.seed,
            stratify=df["strat_key"],
        )

        # Second split: separate val from train (15% of total = 15/85 of remaining)
        adjusted_val_ratio = VAL_RATIO / (1.0 - TEST_RATIO)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_ratio,
            random_state=self.seed,
            stratify=train_val_df["strat_key"],
        )

        # Remove helper columns
        for split_df in [train_df, val_df, test_df]:
            split_df.drop(["age_group", "strat_key"], axis=1, inplace=True)

        logger.info(
            f"Split: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
            f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
            f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
        )

        return train_df, val_df, test_df


def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load, resize to 224×224×3, and normalize an image (Requirement 7.1).

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image array of shape (224, 224, 3), values in [-1, 1],
        or None if loading fails.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = _mobilenetv2_preprocess(image)
    return image


def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply training-time augmentation: random flip, rotation, brightness.

    Args:
        image: Image array of shape (224, 224, 3), values in [-1, 1]

    Returns:
        Augmented image with same shape and value range
    """
    augmented = image.copy()

    # Convert to [0, 255] for OpenCV ops
    augmented = ((augmented + 1.0) * 127.5).astype(np.uint8)

    # Random horizontal flip
    if np.random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # Random rotation ±10 degrees
    angle = np.random.uniform(-10, 10)
    h, w = augmented.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    augmented = cv2.warpAffine(augmented, M, (w, h))

    # Random brightness ±20%
    factor = np.random.uniform(0.8, 1.2)
    augmented = np.clip(augmented.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Convert back to [-1, 1]
    augmented = _mobilenetv2_preprocess(augmented.astype(np.float32))
    return augmented


def create_age_dataset(
    df: pd.DataFrame, batch_size: int = 32, augment: bool = False
):
    """
    Create a tf.data.Dataset for age regression.

    Returns batches of (image, age) pairs.
    """
    paths = df["path"].values
    ages = df["age"].values.astype(np.float32)

    def generator():
        for path, age in zip(paths, ages):
            image = preprocess_image(path)
            if image is None:
                continue
            if augment:
                image = augment_image(image)
            yield image, age

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def create_gender_dataset(
    df: pd.DataFrame, batch_size: int = 32, augment: bool = False
):
    """
    Create a tf.data.Dataset for gender binary classification.

    Returns batches of (image, label) where label: 0=Male, 1=Female.
    """
    paths = df["path"].values
    labels = (df["gender"] == "Female").astype(np.float32).values

    def generator():
        for path, label in zip(paths, labels):
            image = preprocess_image(path)
            if image is None:
                continue
            if augment:
                image = augment_image(image)
            yield image, label

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_age_model() -> "keras.Model":
    """
    Build MobileNetV2-based age regression model (Requirement 7.2).

    Architecture:
    - MobileNetV2 backbone (ImageNet weights, no top, 224×224×3 input)
    - GlobalAveragePooling2D
    - Dense(256, relu)
    - Dropout(0.3)
    - Dense(1, relu) → age output
    """
    input_tensor = layers.Input(shape=(224, 224, 3))
    backbone = MobileNetV2(
        input_tensor=input_tensor,
        weights="imagenet",
        include_top=False,
    )

    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", name="age_dense_256")(x)
    x = layers.Dropout(0.3, name="age_dropout")(x)
    age_output = layers.Dense(1, activation="relu", name="age_output")(x)

    model = keras.Model(inputs=input_tensor, outputs=age_output, name="senior_age_estimator")
    return model


def build_gender_model() -> "keras.Model":
    """
    Build MobileNetV2-based gender binary classifier (Requirement 7.2).

    Architecture:
    - MobileNetV2 backbone (ImageNet weights, no top, 224×224×3 input)
    - GlobalAveragePooling2D
    - Dense(128, relu)
    - Dropout(0.3)
    - Dense(1, sigmoid) → gender output
    """
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", name="gender_dense_128")(x)
    x = layers.Dropout(0.3, name="gender_dropout")(x)
    gender_output = layers.Dense(1, activation="sigmoid", name="gender_output")(x)

    model = keras.Model(
        inputs=base_model.input, outputs=gender_output, name="senior_gender_predictor"
    )
    return model


def train_age_model(
    model: "keras.Model",
    train_ds,
    val_ds,
    epochs: int = 30,
    learning_rate: float = 0.0001,
    fine_tune_from_layer: int = 100,
):
    """
    Train the age estimator model with Huber loss.

    Args:
        model: Built age model
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Max training epochs
        learning_rate: Initial learning rate
        fine_tune_from_layer: Freeze backbone layers before this index

    Returns:
        Training history
    """
    # Freeze early backbone layers for transfer learning
    for layer in model.layers:
        if hasattr(layer, "layers"):  # MobileNetV2 backbone
            for i, sub_layer in enumerate(layer.layers):
                sub_layer.trainable = i >= fine_tune_from_layer

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=Huber(delta=1.0),
        metrics=["mae"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def train_gender_model(
    model: "keras.Model",
    train_ds,
    val_ds,
    epochs: int = 20,
    learning_rate: float = 0.0001,
):
    """
    Train the gender classifier model with binary crossentropy.

    Args:
        model: Built gender model
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Max training epochs
        learning_rate: Initial learning rate

    Returns:
        Training history
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_age_model(
    model: "keras.Model", test_df: pd.DataFrame
) -> float:
    """
    Evaluate age model MAE on test split (Requirement 3.5: MAE ≤ 8 years).

    Returns:
        Mean absolute error on test set
    """
    errors = []
    for _, row in test_df.iterrows():
        image = preprocess_image(row["path"])
        if image is None:
            continue
        image_batch = np.expand_dims(image, axis=0)
        predicted_age = model.predict(image_batch, verbose=0)[0, 0]
        predicted_age = int(np.clip(predicted_age, 1, 100))
        errors.append(abs(predicted_age - row["age"]))

    if not errors:
        return float("inf")
    mae = np.mean(errors)
    return float(mae)


def evaluate_gender_model(
    model: "keras.Model", test_df: pd.DataFrame
) -> float:
    """
    Evaluate gender model accuracy on test split (Requirement 4.4: ≥ 70%).

    Returns:
        Classification accuracy on test set
    """
    correct = 0
    total = 0

    for _, row in test_df.iterrows():
        image = preprocess_image(row["path"])
        if image is None:
            continue
        image_batch = np.expand_dims(image, axis=0)
        prediction = model.predict(image_batch, verbose=0)[0, 0]

        # Sigmoid: < 0.5 → Male, >= 0.5 → Female (matching label encoding)
        predicted_gender = "Female" if prediction >= 0.5 else "Male"
        if predicted_gender == row["gender"]:
            correct += 1
        total += 1

    if total == 0:
        return 0.0
    return correct / total


def main():
    """Main entry point for the senior citizen model training pipeline."""
    if not TENSORFLOW_AVAILABLE:
        logger.error(f"TensorFlow is required but not available: {_tf_import_error}")
        return 1

    parser = argparse.ArgumentParser(
        description="Train age estimator and gender classifier for Senior Citizen Identification"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing UTKFace dataset (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory to save trained model weights (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (auto-generated if not provided)",
    )
    parser.add_argument(
        "--epochs-age",
        type=int,
        default=30,
        help="Number of training epochs for age estimator (default: 30)",
    )
    parser.add_argument(
        "--epochs-gender",
        type=int,
        default=20,
        help="Number of training epochs for gender classifier (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )

    args = parser.parse_args()

    # --- Seed handling (Requirements 7.6, 7.7) ---
    if args.seed is None:
        seed = random.randint(0, 2**31 - 1)
        logger.info(f"No seed provided. Auto-generated seed: {seed}")
        print(f"Auto-generated seed: {seed}")
    else:
        seed = args.seed
        logger.info(f"Using provided seed: {seed}")

    set_seeds(seed)

    # --- Load dataset ---
    logger.info("=" * 60)
    logger.info("Loading UTKFace dataset...")
    logger.info("=" * 60)

    loader = SeniorDatasetLoader(data_dir=args.data_dir, seed=seed)
    df = loader.load()

    # --- Split dataset (70/15/15, stratified) ---
    logger.info("Splitting dataset (70% train, 15% val, 15% test)...")
    train_df, val_df, test_df = loader.split(df)

    # --- Create tf.data datasets ---
    logger.info("Creating training datasets...")
    train_age_ds = create_age_dataset(train_df, batch_size=args.batch_size, augment=True)
    val_age_ds = create_age_dataset(val_df, batch_size=args.batch_size, augment=False)
    train_gender_ds = create_gender_dataset(train_df, batch_size=args.batch_size, augment=True)
    val_gender_ds = create_gender_dataset(val_df, batch_size=args.batch_size, augment=False)

    # --- Train Age Estimator ---
    logger.info("=" * 60)
    logger.info("Training Age Estimator (MobileNetV2 + regression head)...")
    logger.info("=" * 60)

    age_model = build_age_model()
    train_age_model(
        age_model,
        train_age_ds,
        val_age_ds,
        epochs=args.epochs_age,
    )
    logger.info("Age Estimator training complete.")

    # --- Train Gender Classifier ---
    logger.info("=" * 60)
    logger.info("Training Gender Classifier (MobileNetV2 + sigmoid head)...")
    logger.info("=" * 60)

    gender_model = build_gender_model()
    train_gender_model(
        gender_model,
        train_gender_ds,
        val_gender_ds,
        epochs=args.epochs_gender,
    )
    logger.info("Gender Classifier training complete.")

    # --- Evaluate on test split ---
    logger.info("=" * 60)
    logger.info("Evaluating models on test split...")
    logger.info("=" * 60)

    age_mae = evaluate_age_model(age_model, test_df)
    logger.info(f"Age Estimator MAE: {age_mae:.2f} years (target: ≤ 8.0)")
    if age_mae > 8.0:
        logger.warning(
            f"Age MAE {age_mae:.2f} exceeds target of 8.0 years"
        )

    gender_acc = evaluate_gender_model(gender_model, test_df)
    logger.info(f"Gender Classifier Accuracy: {gender_acc:.4f} (target: ≥ 0.70)")
    if gender_acc < 0.70:
        logger.warning(
            f"Gender accuracy {gender_acc:.4f} is below target of 0.70"
        )

    # --- Save models (Requirement 7.4) ---
    logger.info("=" * 60)
    logger.info("Saving model weights...")
    logger.info("=" * 60)

    os.makedirs(args.model_dir, exist_ok=True)

    age_model_path = os.path.join(args.model_dir, DEFAULT_AGE_MODEL_NAME)
    age_model.save(age_model_path)
    logger.info(f"Saved Age Estimator to {age_model_path}")

    gender_model_path = os.path.join(args.model_dir, DEFAULT_GENDER_MODEL_NAME)
    gender_model.save(gender_model_path)
    logger.info(f"Saved Gender Classifier to {gender_model_path}")

    # --- Save training config ---
    config = {
        "seed": seed,
        "dataset_split": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "model_architecture": "MobileNetV2",
        "input_shape": [224, 224, 3],
        "age_model": {
            "output": "regression",
            "loss": "huber",
            "epochs": args.epochs_age,
            "batch_size": args.batch_size,
            "mae_achieved": age_mae,
        },
        "gender_model": {
            "output": "binary_classification",
            "loss": "binary_crossentropy",
            "epochs": args.epochs_gender,
            "batch_size": args.batch_size,
            "accuracy_achieved": gender_acc,
        },
        "no_external_apis": True,  # Requirement 7.8
    }

    config_path = os.path.join(args.model_dir, "senior_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved training config to {config_path}")

    logger.info("=" * 60)
    logger.info("Training pipeline complete.")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
