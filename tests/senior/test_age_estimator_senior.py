"""
Unit tests for SeniorAgeEstimator inference wrapper.

Tests model initialization, prediction output bounds, confidence computation,
input validation, and model load error handling.
"""

import sys
import importlib.machinery
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Mock TensorFlow and Keras before importing the module under test
_tf_mock = MagicMock()
_keras_mock = MagicMock()
_layers_mock = MagicMock()
_mobilenet_mock = MagicMock()

# Set up keras model mock
_model_instance = MagicMock()
_keras_mock.Model.return_value = _model_instance

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

# Patch the TENSORFLOW_AVAILABLE flag and rebuild
import importlib


class TestSeniorAgeEstimatorInit:
    """Tests for SeniorAgeEstimator initialization and model building."""

    def test_input_shape_constant(self):
        """INPUT_SHAPE should be (224, 224, 3)."""
        from src.senior.age_estimator import SeniorAgeEstimator

        assert SeniorAgeEstimator.INPUT_SHAPE == (224, 224, 3)

    def test_init_builds_model(self):
        """Model should be built on initialization."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        assert estimator.model is not None


class TestSeniorAgeEstimatorPredict:
    """Tests for SeniorAgeEstimator.predict() method."""

    def test_predict_returns_tuple(self):
        """predict() should return a tuple of (int, float)."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        # Mock the model predict to return a reasonable age
        estimator.model.predict = MagicMock(return_value=np.array([[45.0]]))

        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        result = estimator.predict(image)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_age_is_integer(self):
        """Age output should be an integer."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        estimator.model.predict = MagicMock(return_value=np.array([[55.7]]))

        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        age, _ = estimator.predict(image)

        assert isinstance(age, int)

    def test_predict_age_clamped_to_1_minimum(self):
        """Age output should be at least 1 even when model outputs < 1."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        estimator.model.predict = MagicMock(return_value=np.array([[0.2]]))

        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        age, _ = estimator.predict(image)

        assert age >= 1

    def test_predict_age_clamped_to_100_maximum(self):
        """Age output should be at most 100 even when model outputs > 100."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        estimator.model.predict = MagicMock(return_value=np.array([[150.0]]))

        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        age, _ = estimator.predict(image)

        assert age <= 100

    def test_predict_age_in_valid_range(self):
        """Age output should be in [1, 100] for normal model output."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        estimator.model.predict = MagicMock(return_value=np.array([[72.3]]))

        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        age, _ = estimator.predict(image)

        assert 1 <= age <= 100
        assert age == 72

    def test_predict_confidence_is_float(self):
        """Confidence output should be a float."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        estimator.model.predict = MagicMock(return_value=np.array([[50.0]]))

        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        _, confidence = estimator.predict(image)

        assert isinstance(confidence, float)

    def test_predict_confidence_in_valid_range(self):
        """Confidence output should be in [0.0, 1.0]."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        estimator.model.predict = MagicMock(return_value=np.array([[50.0]]))

        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        _, confidence = estimator.predict(image)

        assert 0.0 <= confidence <= 1.0

    def test_predict_wrong_shape_raises_error(self):
        """predict() should raise ValueError for wrong image shape."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        wrong_image = np.zeros((100, 100, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected image shape"):
            estimator.predict(wrong_image)

    def test_predict_none_model_raises_error(self):
        """predict() should raise ValueError when model is None."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        estimator.model = None

        image = np.zeros((224, 224, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Model not loaded"):
            estimator.predict(image)


class TestSeniorAgeEstimatorConfidence:
    """Tests for confidence computation logic."""

    def test_confidence_midrange_is_high(self):
        """Predictions in midrange (20-80) should produce higher confidence."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        conf = estimator._compute_confidence(50.0)
        assert conf >= 0.5

    def test_confidence_boundary_is_lower(self):
        """Predictions near boundaries should produce lower confidence."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        conf_low = estimator._compute_confidence(2.0)
        conf_mid = estimator._compute_confidence(50.0)
        assert conf_low < conf_mid

    def test_confidence_below_zero_is_low(self):
        """Raw output below 0 should give low confidence."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        conf = estimator._compute_confidence(-5.0)
        assert conf == 0.1

    def test_confidence_above_hundred_decreases(self):
        """Raw output above 100 should give decreasing confidence."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        conf_101 = estimator._compute_confidence(101.0)
        conf_150 = estimator._compute_confidence(150.0)
        assert conf_101 > conf_150

    def test_confidence_always_in_range(self):
        """Confidence should always be in [0.0, 1.0] regardless of input."""
        from src.senior.age_estimator import SeniorAgeEstimator

        estimator = SeniorAgeEstimator()
        test_values = [-100, -1, 0, 1, 50, 100, 200, 1000]
        for val in test_values:
            conf = estimator._compute_confidence(val)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range for input {val}"


class TestSeniorAgeEstimatorLoad:
    """Tests for model loading and validation."""

    def test_load_missing_file_raises_error(self):
        """load() should raise ModelPredictionError for missing file."""
        from src.senior.age_estimator import SeniorAgeEstimator
        from src.senior.exceptions import ModelPredictionError

        estimator = SeniorAgeEstimator()

        with pytest.raises(ModelPredictionError, match="not found"):
            estimator.load("/nonexistent/path/model.keras")

    def test_load_corrupt_file_raises_error(self, tmp_path):
        """load() should raise ModelPredictionError for corrupt file."""
        from src.senior.age_estimator import SeniorAgeEstimator
        from src.senior.exceptions import ModelPredictionError

        estimator = SeniorAgeEstimator()

        # Create a corrupt file
        corrupt_file = tmp_path / "corrupt_model.keras"
        corrupt_file.write_text("not a real model file")

        # Mock keras.models.load_model to raise an exception for corrupt files
        with patch("src.senior.age_estimator.keras.models.load_model",
                   side_effect=Exception("Invalid file format")):
            with pytest.raises(ModelPredictionError, match="corrupted"):
                estimator.load(str(corrupt_file))
