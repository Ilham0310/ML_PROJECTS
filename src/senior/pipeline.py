"""Pipeline orchestration module for the Senior Citizen Identification system.

Coordinates the full detection pipeline: person detection → crop → age/gender
inference → routing → annotation → logging. Handles frame extraction, error
isolation, reconnection, and graceful shutdown.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np

from src.senior.video_source import VideoSource
from src.senior.person_detector import PersonDetector
from src.senior.crop_preprocessor import CropPreprocessor
from src.senior.age_estimator import SeniorAgeEstimator
from src.senior.gender_predictor import SeniorGenderPredictor
from src.senior.senior_router import SeniorRouter
from src.senior.frame_annotator import FrameAnnotator, AnnotatedDetection
from src.senior.data_logger import DataLogger
from src.senior.models import DetectionRecord

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    """Statistics for the current pipeline session.

    Attributes:
        total_persons: Total number of persons successfully processed.
        total_seniors: Total number of senior citizens identified.
        frames_processed: Total number of frames successfully processed.
    """

    total_persons: int = 0
    total_seniors: int = 0
    frames_processed: int = 0


class Pipeline:
    """Orchestrates the full senior citizen identification pipeline.

    Coordinates person detection, cropping, age/gender inference, classification
    routing, frame annotation, and data logging. Implements per-person error
    isolation, consecutive frame failure tracking with reconnection logic, and
    graceful shutdown.

    Attributes:
        video_source: The video input source (file or webcam).
        detector: Person detection component (YOLOv8-nano).
        preprocessor: Crop and preprocess component.
        age_estimator: Age estimation model wrapper.
        gender_predictor: Gender classification model wrapper.
        router: Senior citizen classification router.
        annotator: Frame annotation component.
        logger_component: Data logging component.
        running: Whether the pipeline is currently processing.
    """

    CONSECUTIVE_FAILURE_THRESHOLD = 50
    RECONNECT_INTERVAL_SEC = 2.0
    RECONNECT_TIMEOUT_SEC = 30.0

    def __init__(
        self,
        video_source: VideoSource,
        detector: PersonDetector,
        preprocessor: CropPreprocessor,
        age_estimator: SeniorAgeEstimator,
        gender_predictor: SeniorGenderPredictor,
        router: SeniorRouter,
        annotator: FrameAnnotator,
        logger_component: DataLogger,
    ):
        """Initialize Pipeline with all component dependencies.

        Args:
            video_source: Video input source (file or webcam).
            detector: Person detector component.
            preprocessor: Crop and preprocess component.
            age_estimator: Age estimation model.
            gender_predictor: Gender classification model.
            router: Senior citizen classification router.
            annotator: Frame annotation component.
            logger_component: Data logging component.
        """
        self.video_source = video_source
        self.detector = detector
        self.preprocessor = preprocessor
        self.age_estimator = age_estimator
        self.gender_predictor = gender_predictor
        self.router = router
        self.annotator = annotator
        self.logger_component = logger_component
        self._running: bool = False
        self._fps: float = 0.0
        self._session_stats: SessionStats = SessionStats()

    @property
    def is_running(self) -> bool:
        """Whether the pipeline is currently processing frames."""
        return self._running

    @property
    def fps(self) -> float:
        """The measured frames per second during processing."""
        return self._fps

    @property
    def session_stats(self) -> SessionStats:
        """Session statistics: total persons, seniors, and frames processed."""
        return self._session_stats

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionRecord]]:
        """Process a single frame through the full pipeline.

        Detects persons, crops and preprocesses each detection, runs age/gender
        inference, applies classification routing, annotates the frame, and
        creates detection records for logging.

        Per-person error isolation: if inference fails for one person, that
        person is skipped and processing continues for remaining detections.

        Args:
            frame: Input BGR image as numpy array (H, W, 3).

        Returns:
            Tuple of (annotated_frame, detection_records):
            - annotated_frame: Frame with bounding boxes and labels drawn.
            - detection_records: List of DetectionRecord for each successfully
              processed person.
        """
        # Detect persons in frame
        detections = self.detector.detect(frame)

        # If no detections, return original frame with empty records
        if not detections:
            return frame.copy(), []

        records: List[DetectionRecord] = []
        annotated_detections: List[AnnotatedDetection] = []
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        for detection in detections:
            try:
                # Crop and preprocess the detected person region
                crop = self.preprocessor.crop_and_preprocess(frame, detection.bbox)

                # Run age estimation
                age, age_conf = self.age_estimator.predict(crop)

                # Run gender classification
                gender, gender_conf = self.gender_predictor.predict(crop)

                # Route through classification logic
                result = self.router.route(age, age_conf, gender, gender_conf)

                # Create detection record
                record = DetectionRecord(
                    timestamp=timestamp,
                    age=result.display_age,
                    gender=result.display_gender,
                    is_senior_citizen="Yes" if result.is_senior else "No",
                )
                records.append(record)

                # Collect annotated detection for frame drawing
                annotated_detections.append((detection, result))

            except Exception as e:
                logger.warning(
                    f"Skipping detection due to error: {e}"
                )
                continue

        # Annotate the frame with all successful detections
        annotated_frame = self.annotator.annotate(frame, annotated_detections)

        # Log all records
        for record in records:
            self.logger_component.log(record)

        # Update session stats
        self._session_stats.frames_processed += 1
        self._session_stats.total_persons += len(records)
        self._session_stats.total_seniors += sum(
            1 for r in records if r.is_senior_citizen == "Yes"
        )

        return annotated_frame, records

    def start(self) -> None:
        """Start the main processing loop.

        Extracts frames from the video source, processes each through the
        pipeline, and handles frame-level errors. Tracks consecutive frame
        failures and triggers reconnection logic when the threshold (50) is
        reached. Measures FPS using time measurements.
        """
        self._running = True
        consecutive_failures = 0
        frame_count = 0
        fps_start_time = time.time()

        if not self.video_source.is_opened():
            if not self.video_source.open():
                logger.error("Failed to open video source.")
                self._running = False
                return

        logger.info("Pipeline started.")

        while self._running:
            try:
                frame = self.video_source.read_frame()

                if frame is None:
                    consecutive_failures += 1
                    logger.warning(
                        f"Frame read failed. Consecutive failures: {consecutive_failures}"
                    )

                    if consecutive_failures >= self.CONSECUTIVE_FAILURE_THRESHOLD:
                        logger.warning(
                            f"Reached {self.CONSECUTIVE_FAILURE_THRESHOLD} consecutive "
                            "frame failures. Attempting reconnection."
                        )
                        reconnected = self._attempt_reconnection()
                        if reconnected:
                            consecutive_failures = 0
                            logger.info("Reconnection successful. Resuming processing.")
                        else:
                            logger.error(
                                "Reconnection failed after "
                                f"{self.RECONNECT_TIMEOUT_SEC}s. Stopping pipeline."
                            )
                            break
                    continue

                # Reset consecutive failure counter on successful read
                consecutive_failures = 0

                # Process the frame
                self.process_frame(frame)

                # Update FPS measurement
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed > 0:
                    self._fps = frame_count / elapsed

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                consecutive_failures += 1

                if consecutive_failures >= self.CONSECUTIVE_FAILURE_THRESHOLD:
                    logger.warning(
                        f"Reached {self.CONSECUTIVE_FAILURE_THRESHOLD} consecutive "
                        "frame failures. Attempting reconnection."
                    )
                    reconnected = self._attempt_reconnection()
                    if reconnected:
                        consecutive_failures = 0
                    else:
                        logger.error(
                            "Reconnection failed. Stopping pipeline."
                        )
                        break

        # Cleanup on loop exit
        self._shutdown()

    def stop(self) -> None:
        """Stop the pipeline gracefully.

        Sets running=False to exit the processing loop, flushes the logger,
        and releases the video source. Must complete within 5 seconds.
        """
        logger.info("Pipeline stop requested.")
        self._running = False
        self._shutdown()

    def _shutdown(self) -> None:
        """Perform shutdown: flush logger, release video source.

        Attempts to flush all pending records and release the video source.
        If flush fails, logs the error with count of unflushed records.
        Enforces a best-effort 5-second shutdown window.
        """
        # Flush pending detection records
        try:
            self.logger_component.flush()
        except Exception as e:
            buffer_count = len(self.logger_component._buffer)
            logger.error(
                f"Flush failed during shutdown: {e}. "
                f"Unflushed records: {buffer_count}"
            )

        # Release video source
        try:
            self.video_source.release()
        except Exception as e:
            logger.error(f"Error releasing video source: {e}")

        logger.info("Pipeline shutdown complete.")

    def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect to the video source.

        Releases the current video source and retries opening it every
        2 seconds for up to 30 seconds.

        Returns:
            True if reconnection was successful, False otherwise.
        """
        # Release current connection
        try:
            self.video_source.release()
        except Exception as e:
            logger.warning(f"Error releasing video source during reconnection: {e}")

        start_time = time.time()
        while time.time() - start_time < self.RECONNECT_TIMEOUT_SEC:
            if not self._running:
                return False

            try:
                if self.video_source.open():
                    return True
            except Exception as e:
                logger.warning(f"Reconnection attempt failed: {e}")

            time.sleep(self.RECONNECT_INTERVAL_SEC)

        # Reconnection timed out - flush and stop
        try:
            self.logger_component.flush()
        except Exception as e:
            logger.error(f"Flush failed after reconnection timeout: {e}")

        return False
