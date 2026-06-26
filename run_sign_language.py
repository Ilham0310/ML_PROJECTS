"""Launch the Sign Language Detection GUI."""

import os
import sys

from PyQt5.QtWidgets import QApplication, QMessageBox

from src.gui.sign_language_window import SignLanguageWindow
from src.sign_language.classifier import SignLanguageClassifier, SignLanguageModelLoadError
from src.sign_language.scheduler import Scheduler


def main() -> int:
    app = QApplication(sys.argv)
    model_path = os.path.join("models", "sign_language_cnn.keras")
    classifier = None
    if os.path.exists(model_path):
        try:
            classifier = SignLanguageClassifier(model_path=model_path)
        except SignLanguageModelLoadError as exc:
            QMessageBox.warning(None, "Model Load Error", str(exc))

    window = SignLanguageWindow(scheduler=Scheduler(), classifier=classifier)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
