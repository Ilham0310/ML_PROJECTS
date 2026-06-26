"""Launch the Age-Emotion Voice Detection GUI."""

import sys

from PyQt5.QtWidgets import QApplication

from src.gui.voice_main_window import VoiceMainWindow
from src.inference.voice_decision_router import VoiceDecisionRouter


def main() -> int:
    app = QApplication(sys.argv)
    window = VoiceMainWindow(router=VoiceDecisionRouter())
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
