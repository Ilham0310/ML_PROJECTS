"""PyQt5 GUI for the Senior Citizen Identification system.

Provides a main window with video display, pipeline controls, source
selection, output configuration, statistics counters, and status indicator.
Runs the processing pipeline in a QThread to avoid blocking the UI.
"""

import logging
import os
import time
from typing import Dict, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.senior.crop_preprocessor import CropPreprocessor
from src.senior.data_logger import DataLogger
from src.senior.file_video_source import FileVideoSource
from src.senior.frame_annotator import FrameAnnotator
from src.senior.models import DetectionRecord
from src.senior.person_detector import PersonDetector
from src.senior.senior_router import SeniorRouter
from src.senior.video_source import VideoSource
from src.senior.webcam_video_source import WebcamVideoSource

logger = logging.getLogger(__name__)

# Supported video file extensions
VIDEO_EXTENSIONS = ("MP4 Files (*.mp4)", "AVI Files (*.avi)", "MOV Files (*.mov)")
VIDEO_FILTER = "Video Files (*.mp4 *.avi *.mov);;All Files (*)"


class PipelineWorker(QThread):
    """QThread worker that runs the processing pipeline in background.

    Emits signals for each processed frame and detection records so
    the GUI can update without blocking.

    Signals:
        frame_ready: Emitted with the annotated frame (numpy array).
        records_ready: Emitted with the list of DetectionRecords for a frame.
        fps_updated: Emitted with the current measured FPS.
        error_occurred: Emitted when a critical error occurs.
        finished_processing: Emitted when the pipeline finishes naturally.
    """

    frame_ready = pyqtSignal(np.ndarray)
    records_ready = pyqtSignal(list)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    finished_processing = pyqtSignal()

    def __init__(self, pipeline_components: Dict, parent=None):
        """Initialize the pipeline worker.

        Args:
            pipeline_components: Dict with pipeline component instances.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self._components = pipeline_components
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the worker is currently processing frames."""
        return self._running

    def stop(self) -> None:
        """Request the worker to stop processing."""
        self._running = False

    def run(self) -> None:
        """Main processing loop executed in the worker thread.

        Reads frames from the video source, processes each through the
        pipeline, and emits signals with results.
        """
        from datetime import datetime

        self._running = True
        video_source = self._components["video_source"]
        detector = self._components["detector"]
        preprocessor = self._components["preprocessor"]
        age_estimator = self._components["age_estimator"]
        gender_predictor = self._components["gender_predictor"]
        router = self._components["router"]
        annotator = self._components["annotator"]
        data_logger = self._components["logger"]

        frame_count = 0
        fps_start_time = time.time()

        try:
            while self._running:
                frame = video_source.read_frame()
                if frame is None:
                    # End of video or read error
                    break

                # Detect persons
                detections = detector.detect(frame)

                records = []
                annotated_detections = []
                timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

                for det in detections:
                    try:
                        crop = preprocessor.crop_and_preprocess(frame, det.bbox)
                        age, age_conf = age_estimator.predict(crop)
                        gender, gender_conf = gender_predictor.predict(crop)
                        result = router.route(age, age_conf, gender, gender_conf)

                        record = DetectionRecord(
                            timestamp=timestamp,
                            age=result.display_age,
                            gender=result.display_gender,
                            is_senior_citizen="Yes" if result.is_senior else "No",
                        )
                        records.append(record)
                        annotated_detections.append((det, result))
                    except Exception as e:
                        logger.warning(f"Skipping detection due to error: {e}")
                        continue

                # Annotate frame
                annotated_frame = annotator.annotate(frame, annotated_detections)

                # Log records
                for record in records:
                    data_logger.log(record)

                # Emit signals
                self.frame_ready.emit(annotated_frame)
                self.records_ready.emit(records)

                # Update FPS measurement
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed > 0:
                    self.fps_updated.emit(frame_count / elapsed)

        except Exception as e:
            logger.error(f"Pipeline worker error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self._running = False
            self.finished_processing.emit()


class SeniorMainWindow(QMainWindow):
    """Main GUI window for the Senior Citizen Identification system.

    Displays annotated video frames, provides controls for starting/stopping
    the pipeline, selecting video sources, configuring output, and shows
    real-time statistics and status.

    Attributes:
        pipeline: Dict of pipeline component instances.
    """

    def __init__(self, pipeline: Optional[Dict] = None, parent=None):
        """Initialize the main window.

        Args:
            pipeline: Optional dict of pre-built pipeline components.
                If None, components will be built from GUI selections.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._pipeline = pipeline
        self._worker: Optional[PipelineWorker] = None
        self._total_persons = 0
        self._total_seniors = 0
        self._current_fps = 0.0

        self._setup_ui()
        self._update_status("Idle")

    def _setup_ui(self) -> None:
        """Build the complete UI layout."""
        self.setWindowTitle("Senior Citizen Identification System")
        self.setMinimumSize(800, 600)

        # Central widget with horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left: Video display area
        self._video_label = QLabel("No video feed")
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.setMinimumSize(480, 360)
        self._video_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; color: #888; "
            "font-size: 16px; border: 1px solid #333; }"
        )
        main_layout.addWidget(self._video_label, stretch=3)

        # Right: Controls and statistics panel
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, stretch=1)

        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self._persons_label = QLabel("Total Persons: 0")
        self._seniors_label = QLabel("Total Seniors: 0")
        self._fps_label = QLabel("Current FPS: 0.0")

        self._persons_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self._seniors_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self._fps_label.setStyleSheet("font-size: 14px;")

        stats_layout.addWidget(self._persons_label)
        stats_layout.addWidget(self._seniors_label)
        stats_layout.addWidget(self._fps_label)
        right_panel.addWidget(stats_group)

        # Video Source group
        source_group = QGroupBox("Video Source")
        source_layout = QVBoxLayout(source_group)

        # Source type selector
        source_type_layout = QHBoxLayout()
        source_type_layout.addWidget(QLabel("Type:"))
        self._source_type_combo = QComboBox()
        self._source_type_combo.addItems(["File", "Webcam"])
        self._source_type_combo.currentTextChanged.connect(self._on_source_type_changed)
        source_type_layout.addWidget(self._source_type_combo)
        source_layout.addLayout(source_type_layout)

        # File path entry with browse button
        file_layout = QHBoxLayout()
        self._file_path_edit = QLineEdit()
        self._file_path_edit.setPlaceholderText("Select video file...")
        self._browse_button = QPushButton("Browse...")
        self._browse_button.clicked.connect(self._browse_video_file)
        file_layout.addWidget(self._file_path_edit)
        file_layout.addWidget(self._browse_button)
        source_layout.addLayout(file_layout)

        # Webcam selector
        webcam_layout = QHBoxLayout()
        webcam_layout.addWidget(QLabel("Camera:"))
        self._webcam_combo = QComboBox()
        self._webcam_combo.addItems([f"Camera {i}" for i in range(10)])
        webcam_layout.addWidget(self._webcam_combo)
        source_layout.addLayout(webcam_layout)

        # Initially show file controls, hide webcam
        self._webcam_combo.setVisible(False)

        right_panel.addWidget(source_group)

        # Output configuration group
        output_group = QGroupBox("Output Configuration")
        output_layout = QVBoxLayout(output_group)

        # Output file path
        out_file_layout = QHBoxLayout()
        self._output_path_edit = QLineEdit()
        self._output_path_edit.setPlaceholderText("Auto-generated if empty")
        self._output_browse_button = QPushButton("Browse...")
        self._output_browse_button.clicked.connect(self._browse_output_file)
        out_file_layout.addWidget(self._output_path_edit)
        out_file_layout.addWidget(self._output_browse_button)
        output_layout.addLayout(out_file_layout)

        # Output format selector
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(["csv", "excel"])
        format_layout.addWidget(self._format_combo)
        output_layout.addLayout(format_layout)

        right_panel.addWidget(output_group)

        # Control buttons
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)

        self._start_button = QPushButton("Start")
        self._start_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-size: 14px; padding: 8px 16px; }"
        )
        self._start_button.clicked.connect(self.start_processing)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; "
            "font-size: 14px; padding: 8px 16px; }"
        )
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop_processing)

        controls_layout.addWidget(self._start_button)
        controls_layout.addWidget(self._stop_button)
        right_panel.addWidget(controls_group)

        # Spacer at the bottom
        right_panel.addStretch()

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

    def _update_status(self, status: str) -> None:
        """Update the status bar text.

        Args:
            status: One of "Processing", "Paused", or "Idle".
        """
        self._status_bar.showMessage(f"Status: {status}")

    def _on_source_type_changed(self, source_type: str) -> None:
        """Handle source type combo box change.

        Args:
            source_type: Either "File" or "Webcam".
        """
        is_file = source_type == "File"
        self._file_path_edit.setVisible(is_file)
        self._browse_button.setVisible(is_file)
        self._webcam_combo.setVisible(not is_file)

    def _browse_video_file(self) -> None:
        """Open a file dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            VIDEO_FILTER,
        )
        if file_path:
            self._file_path_edit.setText(file_path)

    def _browse_output_file(self) -> None:
        """Open a file dialog to select an output file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output File",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)",
        )
        if file_path:
            self._output_path_edit.setText(file_path)

    def _build_video_source(self) -> Optional[VideoSource]:
        """Build a VideoSource from the current GUI selections.

        Returns:
            A VideoSource instance, or None if the source is invalid.
        """
        source_type = self._source_type_combo.currentText()

        if source_type == "File":
            file_path = self._file_path_edit.text().strip()
            if not file_path:
                QMessageBox.warning(
                    self,
                    "Invalid Source",
                    "Please select a video file.",
                )
                return None

            # Validate file existence and format
            if not os.path.isfile(file_path):
                QMessageBox.warning(
                    self,
                    "Invalid Source",
                    f"File not found: {file_path}\n\n"
                    "Please select an existing video file (MP4, AVI, or MOV).",
                )
                return None

            ext = os.path.splitext(file_path)[1].lower()
            if ext not in (".mp4", ".avi", ".mov"):
                QMessageBox.warning(
                    self,
                    "Unsupported Format",
                    f"Unsupported video format: {ext}\n\n"
                    "Accepted formats: MP4, AVI, MOV.",
                )
                return None

            return FileVideoSource(file_path)

        else:
            # Webcam source
            camera_index = self._webcam_combo.currentIndex()
            return WebcamVideoSource(camera_index=camera_index)

    def _build_pipeline_components(self) -> Optional[Dict]:
        """Build pipeline components from the current GUI configuration.

        Returns:
            Dict of pipeline components, or None if source is invalid.
        """
        # If a pre-built pipeline was provided, use it
        if self._pipeline is not None:
            return self._pipeline

        # Build video source from GUI selections
        video_source = self._build_video_source()
        if video_source is None:
            return None

        # Build other components
        yolo_path = os.path.join("models", "yolov8n.pt")
        detector = PersonDetector(yolo_path if os.path.exists(yolo_path) else "yolov8n.pt")
        preprocessor = CropPreprocessor()
        router = SeniorRouter()
        annotator = FrameAnnotator()

        # Output configuration
        output_path = self._output_path_edit.text().strip() or None
        output_format = self._format_combo.currentText()
        data_logger = DataLogger(output_path=output_path, format=output_format)

        # Load model inference components
        from src.senior.age_estimator import SeniorAgeEstimator
        from src.senior.gender_predictor import SeniorGenderPredictor

        age_estimator = SeniorAgeEstimator()
        gender_predictor = SeniorGenderPredictor()

        # Load model weights (default location)
        age_estimator.load(self._first_existing_model_path(
            "models",
            ["senior_age_estimator.keras", "age_estimator.keras"],
        ))
        gender_predictor.load(self._first_existing_model_path(
            "models",
            ["senior_gender_predictor.keras", "gender_predictor.keras"],
        ))

        return {
            "video_source": video_source,
            "detector": detector,
            "preprocessor": preprocessor,
            "age_estimator": age_estimator,
            "gender_predictor": gender_predictor,
            "router": router,
            "annotator": annotator,
            "logger": data_logger,
        }

    def _first_existing_model_path(self, model_dir: str, filenames: list[str]) -> str:
        """Return the first existing model path, or the preferred path for error clarity."""

        for filename in filenames:
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                return path
        return os.path.join(model_dir, filenames[0])

    @pyqtSlot()
    def start_processing(self) -> None:
        """Start the video processing pipeline.

        Opens the video source and starts a PipelineWorker thread.
        Shows an error message if the source cannot be opened.
        """
        if self._worker is not None and self._worker.is_running:
            return

        # Build or retrieve pipeline components
        components = self._build_pipeline_components()
        if components is None:
            return  # Error already displayed to user

        # Open the video source
        video_source = components["video_source"]
        if not video_source.is_opened():
            if not video_source.open():
                QMessageBox.warning(
                    self,
                    "Source Error",
                    "Failed to open video source.\n\n"
                    "The selected file may be corrupted or the camera "
                    "may be unavailable.",
                )
                self._update_status("Idle")
                return

        # Reset counters for new session
        self._total_persons = 0
        self._total_seniors = 0
        self._current_fps = 0.0
        self._update_counters()

        # Create and start worker thread
        self._worker = PipelineWorker(components, parent=self)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.records_ready.connect(self._on_records_ready)
        self._worker.fps_updated.connect(self._on_fps_updated)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished_processing.connect(self._on_finished)
        self._worker.start()

        # Update UI state
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._source_type_combo.setEnabled(False)
        self._file_path_edit.setEnabled(False)
        self._browse_button.setEnabled(False)
        self._webcam_combo.setEnabled(False)
        self._output_path_edit.setEnabled(False)
        self._output_browse_button.setEnabled(False)
        self._format_combo.setEnabled(False)
        self._update_status("Processing")

    @pyqtSlot()
    def stop_processing(self) -> None:
        """Stop the video processing pipeline.

        Stops the worker thread, flushes all pending records, and
        transitions to Idle state.
        """
        if self._worker is None:
            return

        self._update_status("Idle")
        self._worker.stop()

        # Wait for the worker thread to finish (up to 5 seconds)
        if not self._worker.wait(5000):
            logger.warning("Worker thread did not finish within 5 seconds.")
            self._worker.terminate()

        # Flush the logger
        components = self._worker._components
        try:
            data_logger = components["logger"]
            data_logger.flush()
        except Exception as e:
            logger.error(f"Error flushing records on stop: {e}")

        # Release video source
        try:
            video_source = components["video_source"]
            video_source.release()
        except Exception as e:
            logger.error(f"Error releasing video source on stop: {e}")

        self._worker = None
        self._restore_controls()

    @pyqtSlot(np.ndarray)
    def _on_frame_ready(self, frame: np.ndarray) -> None:
        """Handle a new annotated frame from the worker.

        Converts the frame to QPixmap and updates the video display.

        Args:
            frame: Annotated BGR frame as numpy array.
        """
        self.update_frame(frame)

    @pyqtSlot(list)
    def _on_records_ready(self, records: list) -> None:
        """Handle new detection records from the worker.

        Updates the statistics counters.

        Args:
            records: List of DetectionRecord objects.
        """
        self._total_persons += len(records)
        self._total_seniors += sum(
            1 for r in records if r.is_senior_citizen == "Yes"
        )
        self._update_counters()

    @pyqtSlot(float)
    def _on_fps_updated(self, fps: float) -> None:
        """Handle FPS update from the worker.

        Args:
            fps: Current frames per second.
        """
        self._current_fps = fps
        self._fps_label.setText(f"Current FPS: {fps:.1f}")

    @pyqtSlot(str)
    def _on_error(self, error_msg: str) -> None:
        """Handle a pipeline error.

        Args:
            error_msg: Description of the error.
        """
        QMessageBox.critical(
            self,
            "Pipeline Error",
            f"An error occurred during processing:\n\n{error_msg}",
        )
        self.stop_processing()

    @pyqtSlot()
    def _on_finished(self) -> None:
        """Handle natural end of video processing."""
        if self._worker is not None:
            # Flush records
            try:
                data_logger = self._worker._components["logger"]
                data_logger.flush()
            except Exception as e:
                logger.error(f"Error flushing records on finish: {e}")

            # Release video source
            try:
                video_source = self._worker._components["video_source"]
                video_source.release()
            except Exception as e:
                logger.error(f"Error releasing video source on finish: {e}")

            self._worker = None

        self._restore_controls()
        self._update_status("Idle")

    def _restore_controls(self) -> None:
        """Re-enable GUI controls after processing stops."""
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._source_type_combo.setEnabled(True)
        self._file_path_edit.setEnabled(True)
        self._browse_button.setEnabled(True)
        self._webcam_combo.setEnabled(True)
        self._output_path_edit.setEnabled(True)
        self._output_browse_button.setEnabled(True)
        self._format_combo.setEnabled(True)

    def update_frame(self, frame: np.ndarray) -> None:
        """Display an annotated frame in the video widget.

        Converts a BGR numpy array to a QPixmap scaled to fit the
        video display label.

        Args:
            frame: BGR image as numpy array (H, W, 3).
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        # Create QImage and QPixmap
        q_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)

        # Scale to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self._video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled_pixmap)

    def update_counters(self, persons: int, seniors: int) -> None:
        """Update the displayed statistics counters.

        Args:
            persons: Total persons detected.
            seniors: Total senior citizens detected.
        """
        self._total_persons = persons
        self._total_seniors = seniors
        self._update_counters()

    def _update_counters(self) -> None:
        """Refresh the counter labels with current values."""
        self._persons_label.setText(f"Total Persons: {self._total_persons}")
        self._seniors_label.setText(f"Total Seniors: {self._total_seniors}")

    def closeEvent(self, event) -> None:
        """Handle window close: stop processing gracefully.

        Args:
            event: QCloseEvent.
        """
        if self._worker is not None and self._worker.is_running:
            self.stop_processing()
        event.accept()
