"""Shared test fixtures for the Senior Citizen Identification tests."""

import pytest
import numpy as np

from src.senior.models import BoundingBox, Detection, ClassificationResult, DetectionRecord


@pytest.fixture
def sample_bounding_box() -> BoundingBox:
    """A sample bounding box at (100, 50) with size 64x128."""
    return BoundingBox(x=100, y=50, width=64, height=128)


@pytest.fixture
def sample_detection(sample_bounding_box: BoundingBox) -> Detection:
    """A sample detection with 0.85 confidence."""
    return Detection(bbox=sample_bounding_box, confidence=0.85)


@pytest.fixture
def sample_senior_classification() -> ClassificationResult:
    """A sample classification result for a senior citizen."""
    return ClassificationResult(
        is_senior=True,
        is_low_confidence=False,
        display_age=72,
        display_gender="Female",
        box_color=(0, 255, 0),
        label_text="Senior Citizen | Age: 72 | Female",
    )


@pytest.fixture
def sample_detection_record() -> DetectionRecord:
    """A sample detection record for logging."""
    return DetectionRecord(
        timestamp="2024-01-15T14:30:22",
        age=72,
        gender="Female",
        is_senior_citizen="Yes",
    )


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A sample 640x480 BGR frame (black image)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_1080p() -> np.ndarray:
    """A sample 1920x1080 BGR frame (black image)."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)
