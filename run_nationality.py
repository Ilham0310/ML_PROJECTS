"""Launch the Nationality Detection GUI."""

import sys

from PyQt5.QtWidgets import QApplication

from src.gui.nationality_main_window import NationalityMainWindow
from src.nationality.inference_engine import NationalityInferenceEngine


def main() -> int:
    app = QApplication(sys.argv)
    window = NationalityMainWindow(engine=NationalityInferenceEngine())
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
