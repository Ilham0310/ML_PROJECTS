"""Annotation rendering for Car Colour Detection."""

from __future__ import annotations

from typing import Iterable, List, Optional

import cv2
import numpy as np

from src.car_colour.models import AnnotationConfig, BoundingBox


class AnnotationEngine:
    """Draws car rectangles and count overlays."""

    def __init__(self, config: Optional[AnnotationConfig] = None) -> None:
        self.config = config or AnnotationConfig()
        self.config.rect_thickness = max(2, int(self.config.rect_thickness))

    def get_rectangle_colour(self, colour_label: Optional[str]) -> tuple[int, int, int]:
        """Map blue cars to red rectangles and all others to blue rectangles."""

        if isinstance(colour_label, str) and colour_label.lower() == "blue":
            return self.config.blue_car_rect_colour
        return self.config.other_car_rect_colour

    def annotate(
        self,
        image: np.ndarray,
        car_boxes: Iterable[BoundingBox],
        car_colours: Iterable[Optional[str]],
        person_boxes: Iterable[BoundingBox],
    ) -> np.ndarray:
        """Return an annotated image copy."""

        annotated = image.copy()
        boxes = list(car_boxes)
        colours = list(car_colours)
        if len(colours) < len(boxes):
            colours.extend([None] * (len(boxes) - len(colours)))

        for box, colour_label in zip(boxes, colours):
            cv2.rectangle(
                annotated,
                (box.x1, box.y1),
                (box.x2, box.y2),
                self.get_rectangle_colour(colour_label),
                self.config.rect_thickness,
            )
            if colour_label:
                cv2.putText(
                    annotated,
                    str(colour_label),
                    (box.x1, max(15, box.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    self.get_rectangle_colour(colour_label),
                    2,
                    cv2.LINE_AA,
                )

        person_count = len(list(person_boxes))
        overlay = f"Cars: {len(boxes)}  People: {person_count}"
        cv2.putText(
            annotated,
            overlay,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            (255, 255, 255),
            self.config.font_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            overlay,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            (0, 0, 0),
            self.config.font_thickness,
            cv2.LINE_AA,
        )
        return annotated
