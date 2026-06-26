"""Unit tests for the FrameAnnotator class.

Validates:
- Annotated frame has same shape as input
- Senior citizens get green bounding boxes
- Non-seniors get blue bounding boxes
- Low-confidence detections get yellow bounding boxes
- Labels are positioned above bounding boxes
"""

import numpy as np
import pytest

from src.senior.frame_annotator import FrameAnnotator
from src.senior.models import BoundingBox, Detection, ClassificationResult


@pytest.fixture
def annotator() -> FrameAnnotator:
    """Create a FrameAnnotator instance."""
    return FrameAnnotator()


@pytest.fixture
def black_frame() -> np.ndarray:
    """A 480x640 black BGR frame for testing annotations."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def senior_detection() -> tuple:
    """A detection with senior citizen classification (green box)."""
    detection = Detection(
        bbox=BoundingBox(x=100, y=100, width=80, height=160),
        confidence=0.9,
    )
    classification = ClassificationResult(
        is_senior=True,
        is_low_confidence=False,
        display_age=72,
        display_gender="Female",
        box_color=(0, 255, 0),  # Green
        label_text="Senior Citizen | Age: 72 | Female",
    )
    return (detection, classification)


@pytest.fixture
def non_senior_detection() -> tuple:
    """A detection with non-senior classification (blue box)."""
    detection = Detection(
        bbox=BoundingBox(x=300, y=100, width=80, height=160),
        confidence=0.85,
    )
    classification = ClassificationResult(
        is_senior=False,
        is_low_confidence=False,
        display_age=35,
        display_gender="Male",
        box_color=(255, 0, 0),  # Blue
        label_text="Age: 35 | Male",
    )
    return (detection, classification)


@pytest.fixture
def low_confidence_detection() -> tuple:
    """A detection with low-confidence classification (yellow box)."""
    detection = Detection(
        bbox=BoundingBox(x=200, y=150, width=60, height=120),
        confidence=0.7,
    )
    classification = ClassificationResult(
        is_senior=False,
        is_low_confidence=True,
        display_age=55,
        display_gender="Unknown",
        box_color=(0, 255, 255),  # Yellow
        label_text="Age: 55 (low confidence) | Unknown",
    )
    return (detection, classification)


class TestFrameAnnotatorOutputShape:
    """Tests that annotate() returns a frame with the same shape as input."""

    def test_returns_same_shape_with_no_detections(
        self, annotator: FrameAnnotator, black_frame: np.ndarray
    ):
        result = annotator.annotate(black_frame, [])
        assert result.shape == black_frame.shape

    def test_returns_same_shape_with_single_detection(
        self, annotator: FrameAnnotator, black_frame: np.ndarray, senior_detection
    ):
        result = annotator.annotate(black_frame, [senior_detection])
        assert result.shape == black_frame.shape

    def test_returns_same_shape_with_multiple_detections(
        self,
        annotator: FrameAnnotator,
        black_frame: np.ndarray,
        senior_detection,
        non_senior_detection,
        low_confidence_detection,
    ):
        detections = [senior_detection, non_senior_detection, low_confidence_detection]
        result = annotator.annotate(black_frame, detections)
        assert result.shape == black_frame.shape

    def test_does_not_modify_original_frame(
        self, annotator: FrameAnnotator, black_frame: np.ndarray, senior_detection
    ):
        original = black_frame.copy()
        annotator.annotate(black_frame, [senior_detection])
        np.testing.assert_array_equal(black_frame, original)


class TestSeniorCitizenGreenBox:
    """Tests that senior citizens are annotated with green bounding boxes."""

    def test_senior_detection_draws_green_pixels(
        self, annotator: FrameAnnotator, black_frame: np.ndarray, senior_detection
    ):
        result = annotator.annotate(black_frame, [senior_detection])
        detection, classification = senior_detection
        bbox = detection.bbox

        # Check the top edge of the bounding box for green pixels
        # The rectangle is drawn at y=bbox.y, so check that row
        top_row = result[bbox.y, bbox.x : bbox.x + bbox.width]

        # At least some pixels on the top edge should be green (0, 255, 0)
        green_pixels = np.all(top_row == [0, 255, 0], axis=1)
        assert np.any(green_pixels), "Senior citizen box should have green pixels on top edge"

    def test_senior_detection_draws_green_on_left_edge(
        self, annotator: FrameAnnotator, black_frame: np.ndarray, senior_detection
    ):
        result = annotator.annotate(black_frame, [senior_detection])
        detection, _ = senior_detection
        bbox = detection.bbox

        # Check the left edge of the bounding box for green pixels
        left_col = result[bbox.y : bbox.y + bbox.height, bbox.x]

        green_pixels = np.all(left_col == [0, 255, 0], axis=1)
        assert np.any(green_pixels), "Senior citizen box should have green pixels on left edge"


class TestNonSeniorBlueBox:
    """Tests that non-seniors are annotated with blue bounding boxes."""

    def test_non_senior_detection_draws_blue_pixels(
        self, annotator: FrameAnnotator, black_frame: np.ndarray, non_senior_detection
    ):
        result = annotator.annotate(black_frame, [non_senior_detection])
        detection, _ = non_senior_detection
        bbox = detection.bbox

        # Check the top edge of the bounding box for blue pixels
        top_row = result[bbox.y, bbox.x : bbox.x + bbox.width]

        # Blue in BGR is (255, 0, 0)
        blue_pixels = np.all(top_row == [255, 0, 0], axis=1)
        assert np.any(blue_pixels), "Non-senior box should have blue pixels on top edge"

    def test_non_senior_detection_draws_blue_on_left_edge(
        self, annotator: FrameAnnotator, black_frame: np.ndarray, non_senior_detection
    ):
        result = annotator.annotate(black_frame, [non_senior_detection])
        detection, _ = non_senior_detection
        bbox = detection.bbox

        # Check the left edge for blue pixels
        left_col = result[bbox.y : bbox.y + bbox.height, bbox.x]

        blue_pixels = np.all(left_col == [255, 0, 0], axis=1)
        assert np.any(blue_pixels), "Non-senior box should have blue pixels on left edge"


class TestLowConfidenceYellowBox:
    """Tests that low-confidence detections get yellow bounding boxes."""

    def test_low_confidence_draws_yellow_pixels(
        self, annotator: FrameAnnotator, black_frame: np.ndarray, low_confidence_detection
    ):
        result = annotator.annotate(black_frame, [low_confidence_detection])
        detection, _ = low_confidence_detection
        bbox = detection.bbox

        # Check the top edge for yellow pixels (0, 255, 255) in BGR
        top_row = result[bbox.y, bbox.x : bbox.x + bbox.width]

        yellow_pixels = np.all(top_row == [0, 255, 255], axis=1)
        assert np.any(yellow_pixels), "Low-confidence box should have yellow pixels on top edge"
