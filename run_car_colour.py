"""Launch the Car Colour Detection GUI."""

import sys

from PyQt5.QtWidgets import QApplication

from src.gui.car_main_window import CarMainWindow
from src.inference.car_inference_engine import CarInferenceEngine


def main() -> int:
    app = QApplication(sys.argv)
    window = CarMainWindow(engine=CarInferenceEngine())
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
