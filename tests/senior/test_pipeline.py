"""Unit tests for the Pipeline class.

Tests process_frame with mocked components to verify pipeline orchestration,
per-person error isolation, and correct data flow.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.senior.pipeline import Pipeline
from src.senior.models import BoundingBox, Detection, ClassificationResult, DetectionRecord


@pytest.fixture
def mock_video_source():
    """Create a mock VideoSource."""
    source = MagicMock()
    source.is_opened.return_value = True
    source.open.return_value = True
    source.read_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    source.get_fps.return_value = 30.0
    return source


@pytest.fixture
def mock_detector():
    """Create a mock PersonDetector."""
    detector = MagicMock()
    detector.detect.return_value = [
        Detection(bbox=BoundingBox(x=100, y=50, width=80, height=200), confidence=0.9),
        Detection(bbox=BoundingBox(x=300, y=60, width=90, height=180), confidence=0.8),
    ]
    return detector


@pytest.fixture
def mock_preprocessor():
    """Create a mock CropPreprocessor."""
    preprocessor = MagicMock()
    # Returns a valid 224x224x3 normalized array
    preprocessor.crop_and_preprocess.return_value = np.zeros((224, 224, 3), dtype=np.float32)
    return preprocessor


@pytest.fixture
def mock_age_estimator():
    """Create a mock SeniorAgeEstimator."""
    estimator = MagicMock()
    estimator.predict.return_value = (72, 0.85)  # Senior with high confidence
    return estimator


@pytest.fixture
def mock_gender_predictor():
    """Create a mock SeniorGenderPredictor."""
    predictor = MagicMock()
    predictor.predict.return_value = ("Female", 0.92)
    return predictor


@pytest.fixture
def mock_router():
    """Create a mock SeniorRouter."""
    router = MagicMock()
    router.route.return_value = ClassificationResult(
        is_senior=True,
        is_low_confidence=False,
        display_age=72,
        display_gender="Female",
        box_color=(0, 255, 0),
        label_text="Senior Citizen | Age: 72 | Female",
    )
    return router


@pytest.fixture
def mock_annotator():
    """Create a mock FrameAnnotator."""
    annotator = MagicMock()
    # Return a copy of the input frame
    annotator.annotate.side_effect = lambda frame, dets: frame.copy()
    return annotator


@pytest.fixture
def mock_logger():
    """Create a mock DataLogger."""
    data_logger = MagicMock()
    data_logger._buffer = []
    return data_logger


@pytest.fixture
def pipeline(
    mock_video_source,
    mock_detector,
    mock_preprocessor,
    mock_age_estimator,
    mock_gender_predictor,
    mock_router,
    mock_annotator,
    mock_logger,
):
    """Create a Pipeline instance with all mocked components."""
    return Pipeline(
        video_source=mock_video_source,
        detector=mock_detector,
        preprocessor=mock_preprocessor,
        age_estimator=mock_age_estimator,
        gender_predictor=mock_gender_predictor,
        router=mock_router,
        annotator=mock_annotator,
        logger_component=mock_logger,
    )


class TestProcessFrame:
    """Tests for Pipeline.process_frame method."""

    def test_process_frame_returns_annotated_frame_and_records(self, pipeline, mock_detector):
        """process_frame returns an annotated frame and detection records."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated_frame, records = pipeline.process_frame(frame)

        assert annotated_frame is not None
        assert annotated_frame.shape == frame.shape
        assert len(records) == 2  # Two detections from mock

    def test_process_frame_calls_detector(self, pipeline, mock_detector):
        """process_frame invokes the person detector on the input frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)

        mock_detector.detect.assert_called_once_with(frame)

    def test_process_frame_calls_preprocessor_for_each_detection(
        self, pipeline, mock_preprocessor, mock_detector
    ):
        """process_frame crops and preprocesses each detected person."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)

        assert mock_preprocessor.crop_and_preprocess.call_count == 2

    def test_process_frame_calls_age_estimator_for_each_detection(
        self, pipeline, mock_age_estimator
    ):
        """process_frame estimates age for each detected person."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)

        assert mock_age_estimator.predict.call_count == 2

    def test_process_frame_calls_gender_predictor_for_each_detection(
        self, pipeline, mock_gender_predictor
    ):
        """process_frame classifies gender for each detected person."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)

        assert mock_gender_predictor.predict.call_count == 2

    def test_process_frame_calls_router_for_each_detection(self, pipeline, mock_router):
        """process_frame routes each detection through classification."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)

        assert mock_router.route.call_count == 2

    def test_process_frame_logs_records(self, pipeline, mock_logger):
        """process_frame logs a DetectionRecord for each successful detection."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)

        assert mock_logger.log.call_count == 2

    def test_process_frame_detection_record_fields(self, pipeline, mock_router):
        """Detection records contain correct fields from routing."""
        mock_router.route.return_value = ClassificationResult(
            is_senior=True,
            is_low_confidence=False,
            display_age=72,
            display_gender="Female",
            box_color=(0, 255, 0),
            label_text="Senior Citizen | Age: 72 | Female",
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, records = pipeline.process_frame(frame)

        assert records[0].age == 72
        assert records[0].gender == "Female"
        assert records[0].is_senior_citizen == "Yes"
        assert records[0].timestamp  # Non-empty timestamp

    def test_process_frame_no_detections(self, pipeline, mock_detector):
        """process_frame returns empty records when no persons detected."""
        mock_detector.detect.return_value = []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated_frame, records = pipeline.process_frame(frame)

        assert len(records) == 0
        assert annotated_frame.shape == frame.shape

    def test_process_frame_per_person_error_isolation(
        self, pipeline, mock_preprocessor, mock_detector
    ):
        """If processing fails for one person, others are still processed.

        Validates Requirement 10.3: per-person error isolation.
        """
        # First call raises, second call succeeds
        mock_preprocessor.crop_and_preprocess.side_effect = [
            ValueError("Crop failed for person 1"),
            np.zeros((224, 224, 3), dtype=np.float32),
        ]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated_frame, records = pipeline.process_frame(frame)

        # Only one record should be produced (the second person)
        assert len(records) == 1

    def test_process_frame_all_persons_fail(self, pipeline, mock_age_estimator, mock_detector):
        """If all persons fail, returns empty records with annotated frame."""
        mock_age_estimator.predict.side_effect = RuntimeError("Model failure")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated_frame, records = pipeline.process_frame(frame)

        assert len(records) == 0
        assert annotated_frame is not None

    def test_process_frame_annotator_receives_successful_detections(
        self, pipeline, mock_annotator, mock_preprocessor
    ):
        """The annotator only receives successfully processed detections."""
        # First person fails, second succeeds
        mock_preprocessor.crop_and_preprocess.side_effect = [
            ValueError("Crop failed"),
            np.zeros((224, 224, 3), dtype=np.float32),
        ]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)

        # Annotator called with 1 successful detection
        call_args = mock_annotator.annotate.call_args
        annotated_dets = call_args[0][1]
        assert len(annotated_dets) == 1


class TestStop:
    """Tests for Pipeline.stop method."""

    def test_stop_sets_running_false(self, pipeline):
        """stop() sets the running flag to False."""
        pipeline._running = True
        pipeline.stop()
        assert pipeline.is_running is False

    def test_stop_flushes_logger(self, pipeline, mock_logger):
        """stop() flushes the data logger."""
        pipeline._running = True
        pipeline.stop()
        mock_logger.flush.assert_called()

    def test_stop_releases_video_source(self, pipeline, mock_video_source):
        """stop() releases the video source."""
        pipeline._running = True
        pipeline.stop()
        mock_video_source.release.assert_called()

    def test_stop_handles_flush_failure(self, pipeline, mock_logger, mock_video_source):
        """stop() still releases video source even if flush fails."""
        mock_logger.flush.side_effect = IOError("Disk full")
        pipeline._running = True
        pipeline.stop()

        # Video source should still be released
        mock_video_source.release.assert_called()


class TestProperties:
    """Tests for Pipeline properties: is_running, fps, session_stats."""

    def test_is_running_initially_false(self, pipeline):
        """Pipeline.is_running is False before start."""
        assert pipeline.is_running is False

    def test_fps_initially_zero(self, pipeline):
        """Pipeline.fps is 0.0 before any processing."""
        assert pipeline.fps == 0.0

    def test_session_stats_initial(self, pipeline):
        """Pipeline.session_stats starts at zero counts."""
        stats = pipeline.session_stats
        assert stats.total_persons == 0
        assert stats.total_seniors == 0
        assert stats.frames_processed == 0

    def test_session_stats_updates_after_process_frame(self, pipeline, mock_router):
        """session_stats updates after processing a frame with detections."""
        mock_router.route.return_value = ClassificationResult(
            is_senior=True,
            is_low_confidence=False,
            display_age=72,
            display_gender="Female",
            box_color=(0, 255, 0),
            label_text="Senior Citizen | Age: 72 | Female",
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)

        stats = pipeline.session_stats
        assert stats.frames_processed == 1
        assert stats.total_persons == 2  # Two detections from mock
        assert stats.total_seniors == 2  # Both are seniors

    def test_session_stats_non_senior_not_counted(
        self, pipeline, mock_router, mock_detector
    ):
        """Non-seniors are counted as persons but not as seniors."""
        mock_detector.detect.return_value = [
            Detection(bbox=BoundingBox(x=100, y=50, width=80, height=200), confidence=0.9),
        ]
        mock_router.route.return_value = ClassificationResult(
            is_senior=False,
            is_low_confidence=False,
            display_age=35,
            display_gender="Male",
            box_color=(255, 0, 0),
            label_text="Age: 35 | Male",
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)

        stats = pipeline.session_stats
        assert stats.total_persons == 1
        assert stats.total_seniors == 0

    def test_session_stats_accumulates_across_frames(self, pipeline, mock_router):
        """session_stats accumulates across multiple process_frame calls."""
        mock_router.route.return_value = ClassificationResult(
            is_senior=True,
            is_low_confidence=False,
            display_age=72,
            display_gender="Female",
            box_color=(0, 255, 0),
            label_text="Senior Citizen | Age: 72 | Female",
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)
        pipeline.process_frame(frame)

        stats = pipeline.session_stats
        assert stats.frames_processed == 2
        assert stats.total_persons == 4  # 2 per frame × 2 frames
        assert stats.total_seniors == 4


class TestSessionStats:
    """Tests for the SessionStats dataclass."""

    def test_default_values(self):
        """SessionStats initializes with zero counts."""
        from src.senior.pipeline import SessionStats

        stats = SessionStats()
        assert stats.total_persons == 0
        assert stats.total_seniors == 0
        assert stats.frames_processed == 0

    def test_custom_values(self):
        """SessionStats accepts custom initialization values."""
        from src.senior.pipeline import SessionStats

        stats = SessionStats(total_persons=10, total_seniors=3, frames_processed=5)
        assert stats.total_persons == 10
        assert stats.total_seniors == 3
        assert stats.frames_processed == 5
