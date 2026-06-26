"""Unit tests for the PersonDetector class."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.senior.person_detector import PersonDetector
from src.senior.models import BoundingBox, Detection


class TestPersonDetectorInit:
    """Tests for PersonDetector initialization."""

    @patch("src.senior.person_detector.YOLO")
    def test_instantiation_with_defaults(self, mock_yolo):
        """PersonDetector can be instantiated with default parameters."""
        detector = PersonDetector(model_path="yolov8n.pt")

        mock_yolo.assert_called_once_with("yolov8n.pt")
        assert detector.conf_threshold == 0.5
        assert detector.iou_threshold == 0.5
        assert detector.max_detections == 20
        assert detector.min_size == (32, 32)

    @patch("src.senior.person_detector.YOLO")
    def test_instantiation_with_custom_params(self, mock_yolo):
        """PersonDetector respects custom configuration parameters."""
        detector = PersonDetector(
            model_path="custom_model.pt",
            conf_threshold=0.7,
            iou_threshold=0.4,
            max_detections=10,
            min_size=(64, 64),
        )

        mock_yolo.assert_called_once_with("custom_model.pt")
        assert detector.conf_threshold == 0.7
        assert detector.iou_threshold == 0.4
        assert detector.max_detections == 10
        assert detector.min_size == (64, 64)


class TestPersonDetectorDetect:
    """Tests for PersonDetector.detect() method."""

    @patch("src.senior.person_detector.YOLO")
    def test_detect_returns_empty_list_for_blank_frame(self, mock_yolo):
        """detect() returns empty list when no persons are in the frame."""
        # Set up mock to return empty results
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__ = lambda self: 0
        mock_result.boxes.__iter__ = lambda self: iter([])
        mock_model.return_value = [mock_result]

        detector = PersonDetector(model_path="yolov8n.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        assert isinstance(detections, list)
        assert len(detections) == 0

    @patch("src.senior.person_detector.YOLO")
    def test_detect_returns_empty_list_for_none_frame(self, mock_yolo):
        """detect() returns empty list when frame is None."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_path="yolov8n.pt")

        detections = detector.detect(None)

        assert isinstance(detections, list)
        assert len(detections) == 0

    @patch("src.senior.person_detector.YOLO")
    def test_detect_returns_empty_list_for_empty_frame(self, mock_yolo):
        """detect() returns empty list when frame is empty array."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_path="yolov8n.pt")
        frame = np.array([])

        detections = detector.detect(frame)

        assert isinstance(detections, list)
        assert len(detections) == 0

    @patch("src.senior.person_detector.YOLO")
    def test_detect_filters_small_detections(self, mock_yolo):
        """detect() filters out detections smaller than min_size."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Create a mock box that is too small (20x20)
        mock_box_small = MagicMock()
        mock_box_small.xyxy = [MagicMock()]
        mock_box_small.xyxy[0].cpu.return_value.numpy.return_value = np.array([10, 10, 30, 30])
        mock_box_small.conf = [MagicMock()]
        mock_box_small.conf[0].cpu.return_value.numpy.return_value = 0.9

        # Create a mock box that is large enough (100x100)
        mock_box_large = MagicMock()
        mock_box_large.xyxy = [MagicMock()]
        mock_box_large.xyxy[0].cpu.return_value.numpy.return_value = np.array([50, 50, 150, 150])
        mock_box_large.conf = [MagicMock()]
        mock_box_large.conf[0].cpu.return_value.numpy.return_value = 0.85

        mock_result = MagicMock()
        mock_result.boxes = [mock_box_small, mock_box_large]
        mock_model.return_value = [mock_result]

        detector = PersonDetector(model_path="yolov8n.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        assert len(detections) == 1
        assert detections[0].bbox.width == 100
        assert detections[0].bbox.height == 100
        assert detections[0].confidence == 0.85

    @patch("src.senior.person_detector.YOLO")
    def test_detect_sorts_by_confidence_descending(self, mock_yolo):
        """detect() returns detections sorted by confidence (highest first)."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Create mock boxes with different confidences
        boxes = []
        for i, conf in enumerate([0.6, 0.9, 0.75]):
            mock_box = MagicMock()
            mock_box.xyxy = [MagicMock()]
            mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array(
                [i * 100, 0, i * 100 + 50, 50]
            )
            mock_box.conf = [MagicMock()]
            mock_box.conf[0].cpu.return_value.numpy.return_value = conf
            boxes.append(mock_box)

        mock_result = MagicMock()
        mock_result.boxes = boxes
        mock_model.return_value = [mock_result]

        detector = PersonDetector(model_path="yolov8n.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        assert len(detections) == 3
        assert detections[0].confidence == 0.9
        assert detections[1].confidence == 0.75
        assert detections[2].confidence == 0.6

    @patch("src.senior.person_detector.YOLO")
    def test_detect_respects_max_detections(self, mock_yolo):
        """detect() returns at most max_detections results."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Create 5 valid mock boxes
        boxes = []
        for i in range(5):
            mock_box = MagicMock()
            mock_box.xyxy = [MagicMock()]
            mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array(
                [i * 100, 0, i * 100 + 50, 50]
            )
            mock_box.conf = [MagicMock()]
            mock_box.conf[0].cpu.return_value.numpy.return_value = 0.8 - i * 0.05
            boxes.append(mock_box)

        mock_result = MagicMock()
        mock_result.boxes = boxes
        mock_model.return_value = [mock_result]

        detector = PersonDetector(model_path="yolov8n.pt", max_detections=3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        assert len(detections) == 3
        # Should be the top 3 by confidence
        assert detections[0].confidence == pytest.approx(0.8)
        assert detections[1].confidence == pytest.approx(0.75)
        assert detections[2].confidence == pytest.approx(0.7)

    @patch("src.senior.person_detector.YOLO")
    def test_detect_returns_detection_objects(self, mock_yolo):
        """detect() returns properly formed Detection objects with BoundingBox."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([100, 200, 200, 400])
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.92

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        detector = PersonDetector(model_path="yolov8n.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        assert len(detections) == 1
        det = detections[0]
        assert isinstance(det, Detection)
        assert isinstance(det.bbox, BoundingBox)
        assert det.bbox.x == 100
        assert det.bbox.y == 200
        assert det.bbox.width == 100
        assert det.bbox.height == 200
        assert det.confidence == 0.92
