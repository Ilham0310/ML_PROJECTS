"""PyQt5 GUI for Nationality Detection."""

from __future__ import annotations

import os
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.nationality.exceptions import FILE_SIZE_MESSAGE, INVALID_FORMAT_MESSAGE
from src.nationality.inference_engine import (
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
    NationalityInferenceEngine,
)
from src.nationality.prediction_result import PredictionResult


class NationalityMainWindow(QMainWindow):
    """Image-upload GUI for nationality-based conditional predictions."""

    def __init__(
        self,
        engine: Optional[NationalityInferenceEngine] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine or NationalityInferenceEngine()
        self.current_image_path: Optional[str] = None

        self.setWindowTitle("Nationality Detection")
        self.setMinimumSize(860, 540)
        self._setup_ui()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = QVBoxLayout()
        self.image_preview_label = QLabel("No image loaded.")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setFixedSize(400, 400)
        self.image_preview_label.setStyleSheet(
            "QLabel { border: 1px solid #999; background: #f7f7f7; }"
        )
        self.upload_button = QPushButton("Upload Image")
        self.predict_button = QPushButton("Predict")
        self.upload_button.clicked.connect(self._on_upload_clicked)
        self.predict_button.clicked.connect(self._on_predict_clicked)
        left.addWidget(self.image_preview_label)
        left.addWidget(self.upload_button)
        left.addWidget(self.predict_button)
        left.addStretch()
        layout.addLayout(left)

        results_group = QGroupBox("Results")
        form = QFormLayout(results_group)
        self.nationality_label = QLabel("-")
        self.emotion_label = QLabel("-")
        self.age_label = QLabel("-")
        self.dress_colour_label = QLabel("-")
        form.addRow("Nationality:", self.nationality_label)
        form.addRow("Emotion:", self.emotion_label)
        form.addRow("Age:", self.age_label)
        form.addRow("Dress Colour:", self.dress_colour_label)
        layout.addWidget(results_group, stretch=1)
        self.statusBar().showMessage("Ready")

    def _on_upload_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Person Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)",
        )
        if not path:
            return
        if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
            QMessageBox.warning(self, "Unsupported File", INVALID_FORMAT_MESSAGE)
            return
        if os.path.getsize(path) > MAX_FILE_SIZE_BYTES:
            QMessageBox.warning(self, "File Too Large", FILE_SIZE_MESSAGE)
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(
                self,
                "Invalid Image",
                "The selected file could not be opened. Please choose a valid image.",
            )
            return
        self.current_image_path = path
        self.image_preview_label.setPixmap(
            pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _on_predict_clicked(self) -> None:
        if self.current_image_path is None:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")
            return
        self.predict_button.setEnabled(False)
        self.statusBar().showMessage("Processing...")
        try:
            result = self.engine.predict(self.current_image_path)
            self._display_results(result)
            self.statusBar().showMessage("Ready")
        except Exception as exc:
            QMessageBox.warning(self, "Prediction Error", str(exc))
            self.statusBar().showMessage("Ready")
        finally:
            self.predict_button.setEnabled(True)

    def _display_results(self, result: PredictionResult) -> None:
        self.nationality_label.setText(
            f"{result.nationality} ({result.nationality_confidence * 100:.1f}%)"
        )
        self.emotion_label.setText(
            f"{result.emotion} ({result.emotion_confidence * 100:.1f}%)"
        )
        self.age_label.setText("-" if result.age is None else str(result.age))
        if result.dress_colour is None:
            self.dress_colour_label.setText("-")
        else:
            self.dress_colour_label.setText(
                f"{result.dress_colour} ({result.dress_colour_confidence * 100:.1f}%)"
            )
