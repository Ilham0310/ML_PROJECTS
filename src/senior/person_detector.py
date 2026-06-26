"""Person detection using YOLOv8-nano model.

Detects persons in video frames using YOLO inference on COCO class 0 (person),
applies non-maximum suppression, filters by minimum size, and returns top-k
Detection objects sorted by confidence.
"""

from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from src.senior.models import BoundingBox, Detection


class PersonDetector:
    """Detects persons in video frames using YOLOv8-nano.

    Attributes:
        model: The loaded YOLO model instance.
        conf_threshold: Minimum confidence score for detections (default 0.5).
        iou_threshold: IoU threshold for non-maximum suppression (default 0.5).
        max_detections: Maximum number of detections to return (default 20).
        min_size: Minimum bounding box dimensions (width, height) in pixels (default (32, 32)).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_detections: int = 20,
        min_size: Tuple[int, int] = (32, 32),
    ):
        """Initialize PersonDetector with model and filtering parameters.

        Args:
            model_path: Path to the YOLOv8 model weights file.
            conf_threshold: Minimum confidence threshold for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            max_detections: Maximum number of detections to return per frame.
            min_size: Minimum (width, height) in pixels for valid detections.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.min_size = min_size

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect persons in a video frame.

        Runs YOLOv8 inference filtering for person class (COCO class 0),
        applies NMS with the configured IoU threshold, filters detections
        by minimum size, sorts by confidence descending, and returns
        at most max_detections results.

        Args:
            frame: Input image as a numpy array (H, W, C) in BGR format.

        Returns:
            List of Detection objects sorted by confidence (highest first),
            containing at most max_detections items. Returns empty list if
            no persons are detected meeting the criteria.
        """
        if frame is None or frame.size == 0:
            return []

        # Run YOLO inference on person class (class 0) with NMS
        results = self.model(
            frame,
            classes=[0],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Extract bounding box in xyxy format and convert to xywh
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                width = x2 - x1
                height = y2 - y1
                confidence = float(box.conf[0].cpu().numpy())

                # Filter by minimum size
                if width < self.min_size[0] or height < self.min_size[1]:
                    continue

                bbox = BoundingBox(x=x1, y=y1, width=width, height=height)
                detections.append(Detection(bbox=bbox, confidence=confidence))

        # Sort by confidence descending and return top-k
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_detections]
