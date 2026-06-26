"""Core data models for the Senior Citizen Identification system.

Contains dataclasses representing bounding boxes, detections,
classification results, and detection records for logging.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BoundingBox:
    """A rectangular region within a frame that encloses a detected person.

    Attributes:
        x: Top-left x coordinate in pixels.
        y: Top-left y coordinate in pixels.
        width: Box width in pixels.
        height: Box height in pixels.
    """

    x: int
    y: int
    width: int
    height: int


@dataclass
class Detection:
    """A person detection from the YOLOv8-nano model.

    Attributes:
        bbox: The bounding box enclosing the detected person.
        confidence: Detection confidence score in [0.0, 1.0].
    """

    bbox: BoundingBox
    confidence: float


@dataclass
class ClassificationResult:
    """Result of senior citizen classification routing.

    Attributes:
        is_senior: True if age > 60 and confidence >= 0.3.
        is_low_confidence: True if age confidence < 0.3.
        display_age: Age value for display.
        display_gender: Gender string or "Unknown" if low confidence.
        box_color: BGR color tuple for bounding box annotation.
        label_text: Full annotation label string.
    """

    is_senior: bool
    is_low_confidence: bool
    display_age: int
    display_gender: str
    box_color: Tuple[int, int, int]
    label_text: str


@dataclass
class DetectionRecord:
    """A single detection record for data logging.

    Attributes:
        timestamp: ISO 8601 format (YYYY-MM-DDTHH:MM:SS).
        age: Estimated age as integer in [1, 100].
        gender: "Male", "Female", or "Unknown".
        is_senior_citizen: "Yes" or "No".
    """

    timestamp: str
    age: int
    gender: str
    is_senior_citizen: str
