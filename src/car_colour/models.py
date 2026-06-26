"""Shared dataclasses for Car Colour Detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Represents a detected object bounding box in absolute pixels."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 0.0
    class_id: int = -1

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def clipped(self, image_width: int, image_height: int) -> "BoundingBox":
        """Return a box clipped to image bounds."""

        x1 = min(max(0, self.x1), image_width)
        y1 = min(max(0, self.y1), image_height)
        x2 = min(max(0, self.x2), image_width)
        y2 = min(max(0, self.y2), image_height)
        return BoundingBox(
            x1=min(x1, x2),
            y1=min(y1, y2),
            x2=max(x1, x2),
            y2=max(y1, y2),
            confidence=self.confidence,
            class_id=self.class_id,
        )

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class Detection:
    """A single detection with optional car-colour classification."""

    bbox: BoundingBox
    class_id: int
    colour_label: Optional[str] = None


@dataclass
class DetectionResult:
    """Result returned by the object-detection model."""

    car_boxes: List[BoundingBox] = field(default_factory=list)
    person_boxes: List[BoundingBox] = field(default_factory=list)
    car_count: int = 0
    person_count: int = 0


@dataclass
class CarDetectionResult:
    """Complete result from the car colour pipeline."""

    annotated_image: np.ndarray
    car_count: int
    person_count: int
    car_colours: List[str] = field(default_factory=list)


@dataclass
class ColourClassifierConfig:
    """Configuration for the car colour classifier."""

    input_size: Tuple[int, int] = (224, 224)
    num_classes: int = 8
    colour_labels: List[str] = field(
        default_factory=lambda: [
            "black",
            "blue",
            "green",
            "orange",
            "red",
            "silver",
            "white",
            "yellow",
        ]
    )
    weights_path: str = "models/colour_classifier.keras"


@dataclass
class AnnotationConfig:
    """Configuration for rendering car-colour annotations."""

    blue_car_rect_colour: Tuple[int, int, int] = (0, 0, 255)
    other_car_rect_colour: Tuple[int, int, int] = (255, 0, 0)
    rect_thickness: int = 2
    font_scale: float = 0.8
    font_thickness: int = 2
