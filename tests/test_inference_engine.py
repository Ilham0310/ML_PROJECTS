"""Unit tests for InferenceEngine pipeline verification.

Validates task 2.2 criteria:
- File format validation accepts .jpg, .jpeg, .png, .bmp only
- File size validation rejects files > 10 MB
- Face detection uses Haar cascade and enforces exactly 1 face
- Proper exception hierarchy (InferenceError base)
- load_models loads all three models from configured model_dir

Requirements: 1.1, 1.3, 1.4, 2.1, 2.2, 2.3, 10.1, 10.2
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Mock heavy dependencies BEFORE importing the inference engine module.
# This avoids the cv2.dnn.DictValue AttributeError in the test environment.
_cv2_mock = MagicMock()
_cv2_mock.data = MagicMock()
_cv2_mock.data.haarcascades = "/fake/path/"
_cv2_mock.COLOR_BGR2GRAY = 6

_tf_mock = MagicMock()
_modules_to_mock = {
    "cv2": _cv2_mock,
    "tensorflow": _tf_mock,
    "tensorflow.keras": _tf_mock.keras,
    "tensorflow.keras.layers": _tf_mock.keras.layers,
    "tensorflow.keras.applications": _tf_mock.keras.applications,
    "tensorflow.keras.losses": _tf_mock.keras.losses,
    "tensorflow.keras.applications.mobilenet_v2": _tf_mock.keras.applications.mobilenet_v2,
    "tensorflow.keras.models": _tf_mock.keras.models,
}

with patch.dict("sys.modules", _modules_to_mock):
    import src.inference.inference_engine as _ie_module
    from src.inference.inference_engine import (
        InferenceEngine,
        InferenceError,
        ModelLoadError,
        InvalidFileFormatError,
        FileSizeError,
        CorruptImageError,
        NoFaceDetectedError,
        MultipleFacesError,
        MAX_FILE_SIZE_BYTES,
        SUPPORTED_EXTENSIONS,
    )


# ---------------------------------------------------------------------------
# Exception Hierarchy Tests
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    """Confirm proper exception hierarchy: InferenceError as base."""

    def test_inference_error_is_base_exception(self):
        assert issubclass(InferenceError, Exception)

    def test_model_load_error_inherits_inference_error(self):
        assert issubclass(ModelLoadError, InferenceError)

    def test_invalid_file_format_error_inherits_inference_error(self):
        assert issubclass(InvalidFileFormatError, InferenceError)

    def test_file_size_error_inherits_inference_error(self):
        assert issubclass(FileSizeError, InferenceError)

    def test_corrupt_image_error_inherits_inference_error(self):
        assert issubclass(CorruptImageError, InferenceError)

    def test_no_face_detected_error_inherits_inference_error(self):
        assert issubclass(NoFaceDetectedError, InferenceError)

    def test_multiple_faces_error_inherits_inference_error(self):
        assert issubclass(MultipleFacesError, InferenceError)

    def test_all_subclasses_can_be_caught_as_inference_error(self):
        """All specific errors can be caught via the base class."""
        for exc_class in [
            ModelLoadError,
            InvalidFileFormatError,
            FileSizeError,
            CorruptImageError,
            NoFaceDetectedError,
            MultipleFacesError,
        ]:
            try:
                raise exc_class("test message")
            except InferenceError:
                pass  # Expected — caught by base class


# ---------------------------------------------------------------------------
# File Format Validation Tests
# ---------------------------------------------------------------------------

class TestFileFormatValidation:
    """Confirm file format validation accepts .jpg, .jpeg, .png, .bmp only."""

    def _make_engine(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model_dir = "models"
        return engine

    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".bmp"])
    def test_supported_extensions_accepted(self, ext, tmp_path):
        """Supported extensions pass format validation."""
        engine = self._make_engine()
        filepath = str(tmp_path / f"test_image{ext}")
        # Should not raise
        engine._validate_file_format(filepath)

    @pytest.mark.parametrize("ext", [".JPG", ".JPEG", ".PNG", ".BMP", ".Jpg", ".Png"])
    def test_case_insensitive_extensions_accepted(self, ext, tmp_path):
        """Extension check is case-insensitive."""
        engine = self._make_engine()
        filepath = str(tmp_path / f"test_image{ext}")
        engine._validate_file_format(filepath)

    @pytest.mark.parametrize("ext", [".gif", ".tiff", ".webp", ".svg", ".pdf", ".txt", ""])
    def test_unsupported_extensions_rejected(self, ext, tmp_path):
        """Unsupported extensions raise InvalidFileFormatError."""
        engine = self._make_engine()
        filepath = str(tmp_path / f"test_image{ext}")
        with pytest.raises(InvalidFileFormatError):
            engine._validate_file_format(filepath)

    def test_supported_extensions_constant_matches_spec(self):
        """SUPPORTED_EXTENSIONS matches the specification."""
        assert SUPPORTED_EXTENSIONS == {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# File Size Validation Tests
# ---------------------------------------------------------------------------

class TestFileSizeValidation:
    """Confirm file size validation rejects files > 10 MB."""

    def _make_engine(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model_dir = "models"
        return engine

    def test_max_file_size_constant_is_10mb(self):
        """MAX_FILE_SIZE_BYTES is exactly 10 MB."""
        assert MAX_FILE_SIZE_BYTES == 10 * 1024 * 1024

    def test_file_at_10mb_is_accepted(self, tmp_path):
        """A file of exactly 10 MB should pass validation."""
        engine = self._make_engine()
        filepath = tmp_path / "image.jpg"
        filepath.write_bytes(b"\x00" * (10 * 1024 * 1024))
        engine._validate_file_size(str(filepath))

    def test_file_over_10mb_is_rejected(self, tmp_path):
        """A file over 10 MB raises FileSizeError."""
        engine = self._make_engine()
        filepath = tmp_path / "image.jpg"
        # Create file just 1 byte over 10 MB
        filepath.write_bytes(b"\x00" * (10 * 1024 * 1024 + 1))
        with pytest.raises(FileSizeError):
            engine._validate_file_size(str(filepath))

    def test_nonexistent_file_raises_corrupt_image_error(self, tmp_path):
        """A nonexistent path raises CorruptImageError."""
        engine = self._make_engine()
        filepath = str(tmp_path / "nonexistent.jpg")
        with pytest.raises(CorruptImageError):
            engine._validate_file_size(filepath)


# ---------------------------------------------------------------------------
# Face Detection Tests
# ---------------------------------------------------------------------------

class TestFaceDetection:
    """Confirm face detection uses Haar cascade and enforces exactly 1 face."""

    def _make_engine_with_cascade(self, faces_detected):
        """Create engine with mocked cascade returning specified number of faces."""
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model_dir = "models"
        cascade_mock = MagicMock()
        # detectMultiScale returns a numpy array; empty tuple for 0 faces
        if faces_detected == 0:
            cascade_mock.detectMultiScale.return_value = ()
        else:
            cascade_mock.detectMultiScale.return_value = np.array(
                [[10, 10, 50, 50]] * faces_detected
            )
        engine._face_cascade = cascade_mock
        return engine

    def test_zero_faces_raises_no_face_detected_error(self):
        """Zero detected faces raises NoFaceDetectedError."""
        engine = self._make_engine_with_cascade(0)
        fake_image = np.zeros((200, 200, 3), dtype=np.uint8)
        _cv2_mock.cvtColor.return_value = np.zeros((200, 200), dtype=np.uint8)
        with pytest.raises(NoFaceDetectedError):
            engine._detect_faces(fake_image)

    def test_one_face_passes(self):
        """Exactly one detected face does not raise."""
        engine = self._make_engine_with_cascade(1)
        fake_image = np.zeros((200, 200, 3), dtype=np.uint8)
        _cv2_mock.cvtColor.return_value = np.zeros((200, 200), dtype=np.uint8)
        engine._detect_faces(fake_image)  # Should not raise

    def test_multiple_faces_raises_multiple_faces_error(self):
        """More than one detected face raises MultipleFacesError."""
        engine = self._make_engine_with_cascade(3)
        fake_image = np.zeros((200, 200, 3), dtype=np.uint8)
        _cv2_mock.cvtColor.return_value = np.zeros((200, 200), dtype=np.uint8)
        with pytest.raises(MultipleFacesError):
            engine._detect_faces(fake_image)

    def test_haar_cascade_loaded_in_load_models(self, tmp_path):
        """load_models loads the Haar cascade classifier."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        for f in ["age_estimator.keras", "hair_classifier.keras", "gender_predictor.keras"]:
            (model_dir / f).write_text("dummy")

        engine = InferenceEngine(model_dir=str(model_dir))

        # Mock the model classes at module level
        mock_ae = MagicMock()
        mock_hlc = MagicMock()
        mock_gp = MagicMock()
        cascade_instance = MagicMock()
        cascade_instance.empty.return_value = False
        _cv2_mock.CascadeClassifier.return_value = cascade_instance
        _cv2_mock.data.haarcascades = str(tmp_path) + "/"

        orig_ae = _ie_module.AgeEstimator
        orig_hlc = _ie_module.HairLengthClassifier
        orig_gp = _ie_module.GenderPredictor
        try:
            _ie_module.AgeEstimator = mock_ae
            _ie_module.HairLengthClassifier = mock_hlc
            _ie_module.GenderPredictor = mock_gp

            engine.load_models()

            # Verify Haar cascade was loaded
            _cv2_mock.CascadeClassifier.assert_called()
            assert engine._face_cascade is cascade_instance
        finally:
            _ie_module.AgeEstimator = orig_ae
            _ie_module.HairLengthClassifier = orig_hlc
            _ie_module.GenderPredictor = orig_gp


# ---------------------------------------------------------------------------
# Model Loading Tests
# ---------------------------------------------------------------------------

class TestLoadModels:
    """Confirm load_models loads all three models from configured model_dir."""

    def test_loads_all_three_models(self, tmp_path):
        """load_models creates and loads AgeEstimator, HairLengthClassifier, GenderPredictor."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        for f in ["age_estimator.keras", "hair_classifier.keras", "gender_predictor.keras"]:
            (model_dir / f).write_text("dummy")

        engine = InferenceEngine(model_dir=str(model_dir))

        mock_ae = MagicMock()
        mock_hlc = MagicMock()
        mock_gp = MagicMock()
        cascade_instance = MagicMock()
        cascade_instance.empty.return_value = False
        _cv2_mock.CascadeClassifier.return_value = cascade_instance
        _cv2_mock.data.haarcascades = str(tmp_path) + "/"

        orig_ae = _ie_module.AgeEstimator
        orig_hlc = _ie_module.HairLengthClassifier
        orig_gp = _ie_module.GenderPredictor
        try:
            _ie_module.AgeEstimator = mock_ae
            _ie_module.HairLengthClassifier = mock_hlc
            _ie_module.GenderPredictor = mock_gp

            engine.load_models()

            # Verify all three model classes were instantiated
            mock_ae.assert_called_once()
            mock_hlc.assert_called_once()
            mock_gp.assert_called_once()

            # Verify .load() was called on each instance with correct path
            mock_ae.return_value.load.assert_called_once_with(
                os.path.join(str(model_dir), "age_estimator.keras")
            )
            mock_hlc.return_value.load.assert_called_once_with(
                os.path.join(str(model_dir), "hair_classifier.keras")
            )
            mock_gp.return_value.load.assert_called_once_with(
                os.path.join(str(model_dir), "gender_predictor.keras")
            )
        finally:
            _ie_module.AgeEstimator = orig_ae
            _ie_module.HairLengthClassifier = orig_hlc
            _ie_module.GenderPredictor = orig_gp

    def test_missing_models_raise_model_load_error(self, tmp_path):
        """Missing model files raise ModelLoadError."""
        model_dir = tmp_path / "empty_models"
        model_dir.mkdir()

        engine = InferenceEngine(model_dir=str(model_dir))
        with pytest.raises(ModelLoadError):
            engine.load_models()

    def test_model_dir_override_in_load_models(self, tmp_path):
        """model_dir parameter in load_models overrides constructor value."""
        engine = InferenceEngine(model_dir="original_dir")

        model_dir = tmp_path / "new_models"
        model_dir.mkdir()
        for f in ["age_estimator.keras", "hair_classifier.keras", "gender_predictor.keras"]:
            (model_dir / f).write_text("dummy")

        mock_ae = MagicMock()
        mock_hlc = MagicMock()
        mock_gp = MagicMock()
        cascade_instance = MagicMock()
        cascade_instance.empty.return_value = False
        _cv2_mock.CascadeClassifier.return_value = cascade_instance
        _cv2_mock.data.haarcascades = str(tmp_path) + "/"

        orig_ae = _ie_module.AgeEstimator
        orig_hlc = _ie_module.HairLengthClassifier
        orig_gp = _ie_module.GenderPredictor
        try:
            _ie_module.AgeEstimator = mock_ae
            _ie_module.HairLengthClassifier = mock_hlc
            _ie_module.GenderPredictor = mock_gp

            engine.load_models(model_dir=str(model_dir))

            assert engine.model_dir == str(model_dir)
        finally:
            _ie_module.AgeEstimator = orig_ae
            _ie_module.HairLengthClassifier = orig_hlc
            _ie_module.GenderPredictor = orig_gp

    def test_model_files_constant_has_all_three(self):
        """_MODEL_FILES maps all three expected model filenames."""
        expected = {
            "age_estimator": "age_estimator.keras",
            "hair_classifier": "hair_classifier.keras",
            "gender_predictor": "gender_predictor.keras",
        }
        assert InferenceEngine._MODEL_FILES == expected
