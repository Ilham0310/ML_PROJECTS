"""PyQt5 GUI for Sign Language Detection."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.sign_language.classifier import (
    LOW_CONFIDENCE_MESSAGE,
    SignLanguageClassifier,
)
from src.sign_language.hand_detector import HandDetector
from src.sign_language.preprocessor import SignLanguagePreprocessor
from src.sign_language.scheduler import Scheduler

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class SignLanguageWindow(QMainWindow):
    """Main window with image-upload and live-video modes."""

    def __init__(
        self,
        scheduler: Optional[Scheduler] = None,
        hand_detector: Optional[HandDetector] = None,
        preprocessor: Optional[SignLanguagePreprocessor] = None,
        classifier: Optional[SignLanguageClassifier] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.scheduler = scheduler or Scheduler()
        self.hand_detector = hand_detector or HandDetector()
        self.preprocessor = preprocessor or SignLanguagePreprocessor()
        self.classifier = classifier
        self._capture: Optional[cv2.VideoCapture] = None

        self.setWindowTitle("Sign Language Detection")
        self.setMinimumSize(900, 640)
        self._setup_ui()

        self._video_timer = QTimer(self)
        self._video_timer.setInterval(30)
        self._video_timer.timeout.connect(self._process_video_frame)

        self._schedule_timer = QTimer(self)
        self._schedule_timer.setInterval(10_000)
        self._schedule_timer.timeout.connect(self._check_schedule)
        self._schedule_timer.start()
        self._check_schedule()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        self.image_preview_label = QLabel("No image loaded")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumSize(520, 360)
        self.image_preview_label.setStyleSheet(
            "QLabel { border: 1px solid #999; background: #f7f7f7; }"
        )
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self._on_upload_image)
        image_layout.addWidget(self.image_preview_label, stretch=1)
        image_layout.addWidget(self.upload_button)
        self.tabs.addTab(image_tab, "Image Upload")

        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)
        self.video_label = QLabel("No video feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(520, 360)
        self.video_label.setStyleSheet(
            "QLabel { border: 1px solid #999; background: #111; color: #ddd; }"
        )
        buttons = QHBoxLayout()
        self.start_video_button = QPushButton("Start Webcam")
        self.stop_video_button = QPushButton("Stop Webcam")
        self.stop_video_button.setEnabled(False)
        self.start_video_button.clicked.connect(self._on_start_video)
        self.stop_video_button.clicked.connect(self._on_stop_video)
        buttons.addWidget(self.start_video_button)
        buttons.addWidget(self.stop_video_button)
        video_layout.addWidget(self.video_label, stretch=1)
        video_layout.addLayout(buttons)
        self.tabs.addTab(video_tab, "Video")

        self.prediction_label = QLabel("Prediction: -")
        font = self.prediction_label.font()
        font.setPointSize(18)
        font.setBold(True)
        self.prediction_label.setFont(font)
        self.prediction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_label)

    def _check_schedule(self) -> None:
        if self.scheduler.is_operational():
            self._enable_features()
        else:
            self._disable_features()

    def _enable_features(self) -> None:
        self.status_label.setText("")
        self.tabs.setEnabled(True)

    def _disable_features(self) -> None:
        self.status_label.setText(self.scheduler.unavailable_message)
        self.tabs.setEnabled(False)
        self._on_stop_video()

    def _on_upload_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Sign Image",
            "",
            "Images (*.png *.jpg *.jpeg)",
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            QMessageBox.warning(
                self,
                "Unsupported Format",
                "Unsupported file format. Please upload a PNG, JPG, or JPEG image",
            )
            return

        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "Invalid Image", "Unable to read image file")
            return

        self._display_frame(self.image_preview_label, image)
        self._classify_frame(image, no_hand_message="No hand detected in the image")

    def _on_start_video(self) -> None:
        if self._capture is not None:
            return
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            capture.release()
            QMessageBox.warning(
                self,
                "No Webcam",
                "No webcam detected. Please connect a webcam and try again",
            )
            return
        self._capture = capture
        self.start_video_button.setEnabled(False)
        self.stop_video_button.setEnabled(True)
        self._video_timer.start()

    def _on_stop_video(self) -> None:
        if self._video_timer.isActive():
            self._video_timer.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.start_video_button.setEnabled(True)
        self.stop_video_button.setEnabled(False)

    def _process_video_frame(self) -> None:
        if self._capture is None:
            return
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return

        display_frame = frame.copy()
        label = self._classify_frame(frame, no_hand_message=None, show_dialog=False)
        if label:
            cv2.putText(
                display_frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        self._display_frame(self.video_label, display_frame)

    def _classify_frame(
        self,
        frame: np.ndarray,
        no_hand_message: Optional[str],
        show_dialog: bool = True,
    ) -> Optional[str]:
        try:
            hand = self.hand_detector.detect(frame)
        except Exception as exc:
            message = str(exc)
            self.prediction_label.setText(message)
            if show_dialog:
                QMessageBox.warning(self, "Hand Detection Error", message)
            return None

        if hand is None:
            if no_hand_message:
                self.prediction_label.setText(no_hand_message)
                if show_dialog:
                    QMessageBox.information(self, "No Hand", no_hand_message)
            return None

        if self.classifier is None:
            message = "Model not found. Please run training first."
            self.prediction_label.setText(message)
            return None

        processed = self.preprocessor.preprocess(hand)
        label, confidence = self.classifier.predict(processed)
        if label == LOW_CONFIDENCE_MESSAGE:
            display = LOW_CONFIDENCE_MESSAGE
        else:
            display = self.classifier.display_label(label, confidence)
        self.prediction_label.setText(display)
        return display

    def _display_frame(self, target: QLabel, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        q_image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(q_image).scaled(
            target.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        target.setPixmap(pixmap)

    def closeEvent(self, event) -> None:
        self._on_stop_video()
        self.hand_detector.close()
        event.accept()
