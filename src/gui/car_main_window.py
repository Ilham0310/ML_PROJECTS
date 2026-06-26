"""PyQt5 GUI for Car Colour Detection."""

from __future__ import annotations

import os
from typing import Optional

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.inference.car_inference_engine import CarInferenceEngine


class CarMainWindow(QMainWindow):
    """Main application window for car colour detection."""

    def __init__(self, engine: Optional[CarInferenceEngine] = None, parent=None) -> None:
        super().__init__(parent)
        self.engine = engine or CarInferenceEngine()
        self._current_image_path: Optional[str] = None

        self.setWindowTitle("Car Colour Detection")
        self.setMinimumSize(980, 620)
        self._setup_ui()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        button_row = QGridLayout()
        self.upload_button = QPushButton("Upload Image")
        self.detect_button = QPushButton("Detect")
        self.detect_button.setEnabled(False)
        self.upload_button.clicked.connect(self._on_upload_clicked)
        self.detect_button.clicked.connect(self._on_detect_clicked)
        button_row.addWidget(self.upload_button, 0, 0)
        button_row.addWidget(self.detect_button, 0, 1)
        layout.addLayout(button_row)

        image_row = QGridLayout()
        self.original_image_label = QLabel("Original image")
        self.annotated_image_label = QLabel("Annotated image")
        for label in (self.original_image_label, self.annotated_image_label):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(440, 360)
            label.setStyleSheet("QLabel { border: 1px solid #999; background: #f7f7f7; }")
        image_row.addWidget(self.original_image_label, 0, 0)
        image_row.addWidget(self.annotated_image_label, 0, 1)
        layout.addLayout(image_row, stretch=1)

        self.car_count_label = QLabel("Cars: 0")
        self.person_count_label = QLabel("People: 0")
        layout.addWidget(self.car_count_label)
        layout.addWidget(self.person_count_label)

    def _on_upload_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Traffic Image",
            "",
            "Images (*.jpg *.jpeg *.png)",
        )
        if not path:
            return
        if os.path.splitext(path)[1].lower() not in {".jpg", ".jpeg", ".png"}:
            QMessageBox.warning(self, "Unsupported File", "Unsupported image format")
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "Invalid Image", "The selected file is not a valid image")
            return
        self._current_image_path = path
        self._display_image(self.original_image_label, image)
        self.detect_button.setEnabled(True)

    def _on_detect_clicked(self) -> None:
        if self._current_image_path is None:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")
            return
        try:
            result = self.engine.process_image(self._current_image_path)
        except Exception as exc:
            QMessageBox.warning(self, "Detection Error", str(exc))
            return
        self._display_image(self.annotated_image_label, result.annotated_image)
        self.car_count_label.setText(f"Cars: {result.car_count}")
        self.person_count_label.setText(f"People: {result.person_count}")

    def _display_image(self, label: QLabel, image) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        q_image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setPixmap(pixmap)
