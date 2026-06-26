"""Property tests for Sign Language Detection."""

from datetime import time

import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

from src.sign_language.classifier import (
    LOW_CONFIDENCE_MESSAGE,
    format_prediction,
    normalise_probabilities,
)
from src.sign_language.hand_detector import HandBoundingBox, HandDetector
from src.sign_language.preprocessor import SignLanguagePreprocessor
from src.sign_language.scheduler import Scheduler


# Feature: sign-language-detection, Property 1: Scheduler operational window correctness
@given(hour=st.integers(min_value=0, max_value=23), minute=st.integers(min_value=0, max_value=59))
@settings(max_examples=100)
def test_scheduler_operational_window_property(hour, minute):
    scheduler = Scheduler(start_hour=18, end_hour=22)
    assert scheduler.is_operational(time(hour, minute)) is (18 <= hour < 22)


# Feature: sign-language-detection, Property 2: Preprocessing output invariant
@given(
    image=arrays(
        np.uint8,
        st.tuples(
            st.integers(min_value=1, max_value=128),
            st.integers(min_value=1, max_value=128),
            st.just(3),
        ),
    )
)
@settings(max_examples=100, deadline=None)
def test_preprocessing_output_invariant_property(image):
    result = SignLanguagePreprocessor().preprocess(image)
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


# Feature: sign-language-detection, Property 3: Model output is a valid probability distribution
@given(values=st.lists(st.floats(-100, 100, allow_nan=False, allow_infinity=False), min_size=2, max_size=30))
@settings(max_examples=100)
def test_probability_normalisation_property(values):
    probs = normalise_probabilities(values)
    assert probs.shape == (len(values),)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.isclose(float(probs.sum()), 1.0, atol=1e-6)


def _bbox_strategy():
    return st.builds(
        HandBoundingBox,
        x=st.integers(min_value=0, max_value=500),
        y=st.integers(min_value=0, max_value=500),
        width=st.integers(min_value=0, max_value=500),
        height=st.integers(min_value=0, max_value=500),
    )


# Feature: sign-language-detection, Property 4: Largest bounding box selection
@given(boxes=st.lists(_bbox_strategy(), min_size=1, max_size=50))
@settings(max_examples=100)
def test_largest_bounding_box_selection_property(boxes):
    selected = HandDetector.select_largest_bbox(boxes)
    assert selected is not None
    assert all(selected.area >= box.area for box in boxes)


# Feature: sign-language-detection, Property 5: Prediction display formatting
@given(
    label=st.text(min_size=1, max_size=10),
    confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prediction_display_formatting_property(label, confidence):
    display = format_prediction(label, confidence)
    assert label in display
    assert "%" in display


# Feature: sign-language-detection, Property 6: Low confidence thresholding
@given(
    label=st.text(min_size=1, max_size=10),
    confidence=st.floats(min_value=0.0, max_value=0.499999, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_low_confidence_thresholding_property(label, confidence):
    assert format_prediction(label, confidence) == LOW_CONFIDENCE_MESSAGE
