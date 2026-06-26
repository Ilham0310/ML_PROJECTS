"""Inference orchestration for Car Colour Detection."""

from __future__ import annotations

import os
from typing import Optional

import cv2

from src.annotation.annotation_engine import AnnotationEngine
from src.car_colour.models import CarDetectionResult
from src.classification.colour_classifier import ColourClassifier
from src.detection.detection_model import DetectionModel
from src.exceptions import ClassificationError, ModelLoadError

SUPPORTED_CAR_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class CarInferenceEngine:
    """Runs detection, colour classification, and annotation."""

    def __init__(
        self,
        model_dir: str = "models",
        detector: Optional[DetectionModel] = None,
        classifier: Optional[ColourClassifier] = None,
        annotator: Optional[AnnotationEngine] = None,
    ) -> None:
        self.model_dir = model_dir
        self.detector = detector
        self.classifier = classifier
        self.annotator = annotator or AnnotationEngine()

    def load_models(self) -> None:
        """Load YOLOv8 and colour-classifier model weights."""

        try:
            self.detector = DetectionModel(
                weights_path=os.path.join(self.model_dir, "yolov8n.pt")
            )
            self.classifier = ColourClassifier(
                weights_path=os.path.join(self.model_dir, "colour_classifier.keras")
            )
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Model initialization failure: {exc}") from exc

    def process_image(self, image_path: str) -> CarDetectionResult:
        """Process an image path and return annotated output plus counts."""

        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_CAR_IMAGE_EXTENSIONS:
            raise ValueError("The selected file is not a valid image")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("The selected file is not a valid image")

        if self.detector is None or self.classifier is None:
            self.load_models()

        detection_result = self.detector.detect(image)
        car_colours: list[str] = []
        height, width = image.shape[:2]
        for box in detection_result.car_boxes:
            clipped = box.clipped(width, height)
            crop = image[clipped.y1 : clipped.y2, clipped.x1 : clipped.x2]
            try:
                colour = self.classifier.classify(crop)
            except ClassificationError:
                colour = ""
            car_colours.append(colour)

        annotated = self.annotator.annotate(
            image=image,
            car_boxes=detection_result.car_boxes,
            car_colours=car_colours,
            person_boxes=detection_result.person_boxes,
        )
        return CarDetectionResult(
            annotated_image=annotated,
            car_count=detection_result.car_count,
            person_count=detection_result.person_count,
            car_colours=car_colours,
        )
