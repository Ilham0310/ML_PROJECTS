"""PyQt5 GUI for Age-Emotion Voice Detection."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.inference.voice_decision_router import (
    VoiceDecisionRouter,
    VoicePredictionResult,
)


class VoiceWorker(QThread):
    """Background worker that runs voice processing."""

    result_ready = pyqtSignal(object)

    def __init__(self, router: VoiceDecisionRouter, file_path: str, parent=None) -> None:
        super().__init__(parent)
        self.router = router
        self.file_path = file_path

    def run(self) -> None:
        self.result_ready.emit(self.router.process(self.file_path))


class VoiceMainWindow(QMainWindow):
    """Main GUI window for voice analysis."""

    def __init__(self, router: Optional[VoiceDecisionRouter] = None, parent=None) -> None:
        super().__init__(parent)
        self.router = router or VoiceDecisionRouter()
        self._worker: Optional[VoiceWorker] = None
        self.setWindowTitle("Age-Emotion Voice Detection")
        self.setMinimumSize(520, 320)
        self._setup_ui()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.upload_button = QPushButton("Upload Voice Note")
        self.reset_button = QPushButton("Reset")
        self.loading_indicator = QLabel("Processing...")
        self.loading_indicator.setVisible(False)
        self.result_display = QLabel("No voice note uploaded.")
        self.result_display.setWordWrap(True)

        self.upload_button.clicked.connect(self._on_upload_clicked)
        self.reset_button.clicked.connect(self._on_reset_clicked)

        layout.addWidget(self.upload_button)
        layout.addWidget(self.loading_indicator)
        layout.addWidget(self.result_display)
        layout.addWidget(self.reset_button)

    def _on_upload_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Voice Note",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg)",
        )
        if not file_path:
            return
        self.upload_button.setEnabled(False)
        self.loading_indicator.setVisible(True)
        self.result_display.setText("")
        self._worker = VoiceWorker(self.router, file_path, parent=self)
        self._worker.result_ready.connect(self._on_processing_complete)
        self._worker.start()

    def _on_processing_complete(self, result: VoicePredictionResult) -> None:
        self.loading_indicator.setVisible(False)
        self.upload_button.setEnabled(True)
        self.result_display.setText(result.message)
        if result.error:
            QMessageBox.warning(self, "Voice Analysis Error", result.error)

    def _on_reset_clicked(self) -> None:
        self.loading_indicator.setVisible(False)
        self.result_display.setText("No voice note uploaded.")
        self.upload_button.setEnabled(True)
