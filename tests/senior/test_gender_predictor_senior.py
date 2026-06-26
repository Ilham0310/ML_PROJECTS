"""
Unit tests for SeniorGenderPredictor inference wrapper.

Tests model initialization, prediction output labels, confidence bounds,
"Unknown" threshold behavior, input validation, and model load error handling.
"""

import sys
import importlib.machinery
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock TensorFlow and Keras before importing the module under test
for mod_name in [
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.applications",
    "tensorflow.keras.applications.MobileNetV2",
    "tensorflow.keras.layers",
    "tensorflow.keras.models",
]:
    if mod_name not in sys.modules:
        module_mock = MagicMock()
        module_mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, loader=None)
        sys.modules[mod_name] = module_mock


class TestSeniorGenderPredictorInit:
    """Tests for SeniorGenderPredictor initialization and model building."""

    def test_input_shape_constant(self):
        """INPUT_SHAPE should be (224, 224, 3)."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        assert SeniorGenderPredictor.INPUT_SHAPE == (224, 224, 3)

    def test_unknown_threshold_constant(self):
        """UNKNOWN_CONFIDENCE_THRESHOLD should be 0.4."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        assert SeniorGenderPredictor.UNKNOWN_CONFIDENCE_THRESHOLD == 0.4

    def test_init_builds_model(self):
        """Model should be built on initialization."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        assert predictor.model is not None


class TestSeniorGenderPredictorPredict:
    """Tests for SeniorGenderPredictor.predict() method."""

    def test_predict_returns_tuple(self):
        """predict() should return a tuple of (str, float)."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        predictor.model.predict = MagicMock(return_value=np.array([[0.85]]))

        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        result = predictor.predict(image)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_label_is_string(self):
        """Gender label should be a string."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        predictor.model.predict = MagicMock(return_value=np.array([[0.85]]))

        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        label, _ = predictor.predict(image)

        assert isinstance(label, str)

    def test_predict_high_sigmoid_returns_male(self):
        """Sigmoid >= 0.5 should return Male."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        predictor.model.predict = MagicMock(return_value=np.array([[0.9]]))

        image = np.zeros((224, 224, 3), dtype=np.float32)
        label, confidence = predictor.predict(image)

        assert label == "Male"
        assert confidence == pytest.approx(0.9)

    def test_predict_low_sigmoid_returns_female(self):
        """Sigmoid < 0.5 should return Female."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        predictor.model.predict = MagicMock(return_value=np.array([[0.1]]))

        image = np.zeros((224, 224, 3), dtype=np.float32)
        label, confidence = predictor.predict(image)

        assert label == "Female"
        assert confidence == pytest.approx(0.9)

    def test_predict_confidence_is_float(self):
        """Confidence output should be a float."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        predictor.model.predict = MagicMock(return_value=np.array([[0.7]]))

        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        _, confidence = predictor.predict(image)

        assert isinstance(confidence, float)

    def test_predict_confidence_in_valid_range(self):
        """Confidence output should be in [0.0, 1.0]."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        predictor.model.predict = MagicMock(return_value=np.array([[0.8]]))

        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        _, confidence = predictor.predict(image)

        assert 0.0 <= confidence <= 1.0

    def test_predict_wrong_shape_raises_error(self):
        """predict() should raise ValueError for wrong image shape."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        wrong_image = np.zeros((100, 100, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected image shape"):
            predictor.predict(wrong_image)

    def test_predict_none_model_raises_error(self):
        """predict() should raise ValueError when model is None."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        predictor.model = None

        image = np.zeros((224, 224, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Model not loaded"):
            predictor.predict(image)


class TestSeniorGenderPredictorUnknownThreshold:
    """Tests for 'Unknown' gender handling when confidence < 0.4."""

    def test_sigmoid_near_half_returns_label_not_unknown(self):
        """Sigmoid at 0.5 gives confidence 0.5, which is >= 0.4, so not Unknown."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        predictor.model.predict = MagicMock(return_value=np.array([[0.5]]))

        image = np.zeros((224, 224, 3), dtype=np.float32)
        label, confidence = predictor.predict(image)

        # confidence = max(0.5, 0.5) = 0.5, which is >= 0.4
        assert label in {"Male", "Female"}
        assert confidence >= 0.4

    def test_high_confidence_male(self):
        """High sigmoid output (0.9) should give Male with high confidence."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        predictor.model.predict = MagicMock(return_value=np.array([[0.9]]))

        image = np.zeros((224, 224, 3), dtype=np.float32)
        label, confidence = predictor.predict(image)

        assert label == "Male"
        assert confidence == pytest.approx(0.9)

    def test_high_confidence_female(self):
        """Low sigmoid output (0.1) should give Female with 0.9 confidence."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        predictor.model.predict = MagicMock(return_value=np.array([[0.1]]))

        image = np.zeros((224, 224, 3), dtype=np.float32)
        label, confidence = predictor.predict(image)

        assert label == "Female"
        assert confidence == pytest.approx(0.9)

    def test_confidence_at_threshold_boundary(self):
        """Sigmoid output 0.6 gives confidence 0.6, which is >= 0.4, so returns label."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()
        predictor.model.predict = MagicMock(return_value=np.array([[0.6]]))

        image = np.zeros((224, 224, 3), dtype=np.float32)
        label, confidence = predictor.predict(image)

        assert label == "Male"
        assert confidence == pytest.approx(0.6)

    def test_label_is_always_valid_value(self):
        """Label should always be one of Male, Female, or Unknown."""
        from src.senior.gender_predictor import SeniorGenderPredictor

        predictor = SeniorGenderPredictor()

        # Test various sigmoid outputs
        for sigmoid_val in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            predictor.model.predict = MagicMock(
                return_value=np.array([[sigmoid_val]])
            )
            image = np.zeros((224, 224, 3), dtype=np.float32)
            label, confidence = predictor.predict(image)

            assert label in {"Male", "Female", "Unknown"}, (
                f"Unexpected label '{label}' for sigmoid={sigmoid_val}"
            )
            assert 0.0 <= confidence <= 1.0


class TestSeniorGenderPredictorLoad:
    """Tests for model loading and validation."""

    def test_load_missing_file_raises_error(self):
        """load() should raise ModelPredictionError for missing file."""
        from src.senior.gender_predictor import SeniorGenderPredictor
        from src.senior.exceptions import ModelPredictionError

        predictor = SeniorGenderPredictor()

        with pytest.raises(ModelPredictionError, match="not found"):
            predictor.load("/nonexistent/path/model.keras")

    def test_load_corrupt_file_raises_error(self, tmp_path):
        """load() should raise ModelPredictionError for corrupt file."""
        from src.senior.gender_predictor import SeniorGenderPredictor
        from src.senior.exceptions import ModelPredictionError

        predictor = SeniorGenderPredictor()

        # Create a corrupt file
        corrupt_file = tmp_path / "corrupt_model.keras"
        corrupt_file.write_text("not a real model file")

        # Mock keras.models.load_model to raise an exception for corrupt files
        with patch("src.senior.gender_predictor.keras.models.load_model",
                   side_effect=Exception("Invalid file format")):
            with pytest.raises(ModelPredictionError, match="corrupted"):
                predictor.load(str(corrupt_file))
