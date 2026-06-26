"""YOLOv8 detection wrapper for cars and people."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from src.car_colour.models import BoundingBox, DetectionResult
from src.exceptions import ModelLoadError

CAR_CLASS_ID = 2
PERSON_CLASS_ID = 0


class DetectionModel:
    """YOLOv8-based object detector for COCO car and person classes."""

    def __init__(
        self,
        weights_path: str = "models/yolov8n.pt",
        model: Optional[object] = None,
    ) -> None:
        self.weights_path = weights_path
        self.model = model
        if model is not None:
            return
        if not os.path.isfile(weights_path):
            raise ModelLoadError(
                f"Detection model failed to load: weights not found at {weights_path}"
            )
        try:
            from ultralytics import YOLO

            self.model = YOLO(weights_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Detection model failed to load from {weights_path}: {exc}"
            ) from exc

    @staticmethod
    def _box_from_values(values, confidence: float, class_id: int) -> BoundingBox:
        x1, y1, x2, y2 = [int(round(float(v))) for v in values[:4]]
        return BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=float(confidence),
            class_id=int(class_id),
        )

    @staticmethod
    def _parse_ultralytics_result(result) -> List[BoundingBox]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = np.asarray(getattr(boxes, "xyxy", []), dtype=float)
        confidences = np.asarray(getattr(boxes, "conf", []), dtype=float)
        classes = np.asarray(getattr(boxes, "cls", []), dtype=int)
        parsed: List[BoundingBox] = []
        for values, confidence, class_id in zip(xyxy, confidences, classes):
            parsed.append(
                DetectionModel._box_from_values(values, confidence, int(class_id))
            )
        return parsed

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run object detection and return only car/person detections."""

        if self.model is None:
            raise ModelLoadError("Detection model has not been loaded.")
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")

        raw_results = self.model(image)
        if raw_results is None:
            raw_results = []
        if not isinstance(raw_results, (list, tuple)):
            raw_results = [raw_results]

        all_boxes: List[BoundingBox] = []
        for result in raw_results:
            all_boxes.extend(self._parse_ultralytics_result(result))

        height, width = image.shape[:2]
        car_boxes = [
            box.clipped(width, height)
            for box in all_boxes
            if box.class_id == CAR_CLASS_ID
        ]
        person_boxes = [
            box.clipped(width, height)
            for box in all_boxes
            if box.class_id == PERSON_CLASS_ID
        ]
        return DetectionResult(
            car_boxes=car_boxes,
            person_boxes=person_boxes,
            car_count=len(car_boxes),
            person_count=len(person_boxes),
        )
