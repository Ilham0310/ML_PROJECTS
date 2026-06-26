"""Property tests for Nationality Detection."""

from unittest.mock import MagicMock

import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

from src.nationality.age_estimator import NationalityAgeEstimator
from src.nationality.decision_router import NationalityDecisionRouter
from src.nationality.dress_colour_classifier import DressColourClassifier
from src.nationality.emotion_predictor import EmotionPredictor
from src.nationality.exceptions import INVALID_FORMAT_MESSAGE, InvalidFileFormatError
from src.nationality.inference_engine import NationalityInferenceEngine, SUPPORTED_EXTENSIONS
from src.nationality.nationality_detector import NationalityDetector
from src.nationality.preprocessor import NationalityPreprocessor


# Feature: nationality-detection, Property 1: Unsupported file formats are rejected
@given(ext=st.text(min_size=0, max_size=8).filter(lambda value: f".{value.lower().lstrip('.')}" not in SUPPORTED_EXTENSIONS))
@settings(max_examples=100)
def test_unsupported_file_formats_are_rejected_property(ext):
    engine = NationalityInferenceEngine.__new__(NationalityInferenceEngine)
    suffix = ext if ext.startswith(".") else f".{ext}"
    with np.testing.assert_raises(InvalidFileFormatError) as exc_info:
        NationalityInferenceEngine._validate_file_format(engine, f"image{suffix}")
    assert str(exc_info.exception) == INVALID_FORMAT_MESSAGE


# Feature: nationality-detection, Property 2: Nationality prediction produces valid output
@given(image=arrays(np.float32, (128, 128, 3), elements=st.floats(0, 1, allow_nan=False, allow_infinity=False)))
@settings(max_examples=100, deadline=None)
def test_nationality_prediction_valid_output_property(image):
    label, confidence = NationalityDetector().predict(image)
    assert label in NationalityDetector.CLASSES
    assert 0.0 <= confidence <= 1.0


# Feature: nationality-detection, Property 3: Emotion prediction produces valid output
@given(image=arrays(np.float32, (48, 48, 1), elements=st.floats(0, 1, allow_nan=False, allow_infinity=False)))
@settings(max_examples=100, deadline=None)
def test_emotion_prediction_valid_output_property(image):
    label, confidence = EmotionPredictor().predict(image)
    assert label in EmotionPredictor.CLASSES
    assert 0.0 <= confidence <= 1.0


# Feature: nationality-detection, Property 4: Decision Router produces correct conditional outputs
@given(nationality=st.sampled_from(NationalityDetector.CLASSES))
@settings(max_examples=100)
def test_decision_router_conditional_outputs_property(nationality):
    router = NationalityDecisionRouter()
    face = np.zeros((128, 128, 3), dtype=np.float32)
    full = np.zeros((128, 128, 3), dtype=np.float32)
    emotion = MagicMock()
    emotion.predict.return_value = ("happy", 0.8)
    age = MagicMock()
    age.predict.return_value = 35
    dress = MagicMock()
    dress.predict.return_value = ("blue", 0.7)

    result = router.route(face, full, nationality, 0.9, emotion, age, dress)

    assert result.nationality == nationality
    assert result.emotion == "happy"
    assert (result.age is not None) is (nationality in {"Indian", "US/American"})
    assert (result.dress_colour is not None) is (nationality in {"Indian", "African"})


# Feature: nationality-detection, Property 5: Image preprocessing produces correctly shaped and normalised output
@given(
    image=arrays(
        np.uint8,
        st.tuples(
            st.integers(1, 160),
            st.integers(1, 160),
            st.just(3),
        ),
    )
)
@settings(max_examples=100, deadline=None)
def test_nationality_preprocessing_shape_and_range_property(image):
    preprocessor = NationalityPreprocessor()
    colour = preprocessor.preprocess_face(image, (128, 128))
    gray = preprocessor.preprocess_face_grayscale(image, (48, 48))
    full = preprocessor.preprocess_full_image(image, (128, 128))
    assert colour.shape == (128, 128, 3)
    assert gray.shape == (48, 48, 1)
    assert full.shape == (128, 128, 3)
    for output in (colour, gray, full):
        assert output.dtype == np.float32
        assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0


@given(image=arrays(np.float32, (128, 128, 3), elements=st.floats(0, 1, allow_nan=False, allow_infinity=False)))
@settings(max_examples=50, deadline=None)
def test_nationality_age_estimator_output_range_property(image):
    age = NationalityAgeEstimator().predict(image)
    assert isinstance(age, int)
    assert 1 <= age <= 120
