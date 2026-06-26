"""Senior Citizen Identification application entry point.

Provides argument parsing, pipeline construction, and dual-mode
operation (GUI/CLI) for the senior citizen identification system.
"""

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from src.senior.crop_preprocessor import CropPreprocessor
from src.senior.data_logger import DataLogger
from src.senior.file_video_source import FileVideoSource
from src.senior.frame_annotator import FrameAnnotator
from src.senior.person_detector import PersonDetector
from src.senior.senior_router import SeniorRouter
from src.senior.webcam_video_source import WebcamVideoSource

logger = logging.getLogger(__name__)

# Video file extensions that indicate a file source (not a camera index)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")


class SeniorCitizenApp:
    """Application entry point for the Senior Citizen Identification system.

    Parses command-line arguments, builds the processing pipeline, and
    runs in either GUI mode (PyQt5) or CLI mode (headless processing).

    Usage:
        app = SeniorCitizenApp()
        app.run()
    """

    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse command-line arguments.

        Args:
            args: Optional list of argument strings (for testing).
                If None, uses sys.argv.

        Returns:
            Parsed argparse.Namespace with:
                - source: file path (str) or camera index (int)
                - output: output file path or None
                - format: "csv" or "excel"
                - gui: True if GUI mode, False for CLI
                - model_dir: path to model weights directory
        """
        parser = argparse.ArgumentParser(
            description="Senior Citizen Identification System - "
            "Detect and identify senior citizens in video feeds."
        )

        parser.add_argument(
            "--source",
            type=str,
            default="0",
            help="Video source: file path (.mp4/.avi/.mov) or camera index (0-10). "
            "Default: 0 (default camera).",
        )

        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output file path for detection records. "
            "Default: auto-generated filename in current directory.",
        )

        parser.add_argument(
            "--format",
            type=str,
            choices=["csv", "excel"],
            default="csv",
            help="Output file format. Default: csv.",
        )

        parser.add_argument(
            "--gui",
            action="store_true",
            default=False,
            help="Launch PyQt5 GUI mode. Without this flag, runs in CLI mode.",
        )

        parser.add_argument(
            "--model-dir",
            type=str,
            default="models",
            help="Directory containing model weight files. Default: models.",
        )

        parsed = parser.parse_args(args)

        # Convert source to appropriate type
        parsed.source = self._parse_source(parsed.source)

        return parsed

    def _parse_source(self, source_str: str) -> object:
        """Parse the --source argument into a file path or camera index.

        A source is treated as a file path if it ends with a video extension
        (.mp4, .avi, .mov). Otherwise, it's treated as a camera index integer.

        Args:
            source_str: Raw source string from command line.

        Returns:
            str (file path) if the source ends with a video extension,
            int (camera index) if the source is a valid integer 0-10.

        Raises:
            SystemExit: If the source is neither a valid file path
                nor a valid camera index.
        """
        # Check if it looks like a video file path
        if source_str.lower().endswith(VIDEO_EXTENSIONS):
            return source_str

        # Try to parse as camera index
        try:
            index = int(source_str)
            if 0 <= index <= 10:
                return index
            else:
                print(
                    f"Error: Camera index must be between 0 and 10, got {index}.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except ValueError:
            print(
                f"Error: '{source_str}' is not a valid video file path "
                f"(.mp4/.avi/.mov) or camera index (0-10).",
                file=sys.stderr,
            )
            sys.exit(1)

    def _build_pipeline(self, args: argparse.Namespace):
        """Instantiate all pipeline components from parsed arguments.

        Args:
            args: Parsed command-line arguments.

        Returns:
            A dict of pipeline components ready for processing.
        """
        # Build video source
        if isinstance(args.source, str):
            video_source = FileVideoSource(args.source)
        else:
            video_source = WebcamVideoSource(camera_index=args.source)

        # Build processing components
        import os

        yolo_model_path = os.path.join(args.model_dir, "yolov8n.pt")
        detector = PersonDetector(yolo_model_path if os.path.exists(yolo_model_path) else "yolov8n.pt")
        preprocessor = CropPreprocessor()
        router = SeniorRouter()
        annotator = FrameAnnotator()
        data_logger = DataLogger(output_path=args.output, format=args.format)

        # Build model inference components
        from src.senior.age_estimator import SeniorAgeEstimator
        from src.senior.gender_predictor import SeniorGenderPredictor

        age_estimator = SeniorAgeEstimator()
        gender_predictor = SeniorGenderPredictor()

        age_model_path = self._first_existing_model_path(
            args.model_dir,
            ["senior_age_estimator.keras", "age_estimator.keras"],
        )
        gender_model_path = self._first_existing_model_path(
            args.model_dir,
            ["senior_gender_predictor.keras", "gender_predictor.keras"],
        )
        age_estimator.load(age_model_path)
        gender_predictor.load(gender_model_path)

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

        import os

        for filename in filenames:
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                return path
        return os.path.join(model_dir, filenames[0])

    def _run_cli(self, pipeline: dict) -> None:
        """Run the system in headless CLI mode.

        Processes video frames and logs detection records to file.
        Handles Ctrl+C gracefully: flushes records, releases source,
        and exits within 5 seconds.

        Args:
            pipeline: Dict of pipeline components from _build_pipeline().
        """
        video_source = pipeline["video_source"]
        detector = pipeline["detector"]
        preprocessor = pipeline["preprocessor"]
        age_estimator = pipeline["age_estimator"]
        gender_predictor = pipeline["gender_predictor"]
        router = pipeline["router"]
        data_logger = pipeline["logger"]

        running = True
        shutdown_requested = False

        def _signal_handler(signum, frame):
            nonlocal running, shutdown_requested
            if shutdown_requested:
                # Force exit on second Ctrl+C
                sys.exit(1)
            shutdown_requested = True
            running = False
            logger.info("Shutdown requested. Flushing records and releasing resources...")

        # Register signal handler for graceful Ctrl+C
        signal.signal(signal.SIGINT, _signal_handler)

        try:
            # Open video source
            if not video_source.open():
                print("Error: Failed to open video source.", file=sys.stderr)
                return

            logger.info("CLI mode started. Processing video. Press Ctrl+C to stop.")
            print("Processing video... Press Ctrl+C to stop.")

            from datetime import datetime
            from src.senior.models import DetectionRecord

            while running:
                frame = video_source.read_frame()
                if frame is None:
                    # End of video or read error
                    break

                # Detect persons
                detections = detector.detect(frame)
                if not detections:
                    continue

                # Process each detection
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
                        data_logger.log(record)
                    except Exception as e:
                        logger.warning(f"Skipping detection due to error: {e}")
                        continue

        finally:
            # Graceful shutdown within 5 seconds
            shutdown_start = time.time()
            try:
                data_logger.flush()
                data_logger.close()
            except Exception as e:
                logger.error(f"Error flushing records during shutdown: {e}")

            try:
                video_source.release()
            except Exception as e:
                logger.error(f"Error releasing video source: {e}")

            elapsed = time.time() - shutdown_start
            logger.info(f"Shutdown completed in {elapsed:.2f}s.")
            print("Processing stopped.")

    def _run_gui(self, pipeline: dict) -> None:
        """Run the system in PyQt5 GUI mode.

        Args:
            pipeline: Dict of pipeline components from _build_pipeline().
        """
        try:
            from src.senior.gui import SeniorMainWindow
        except ImportError:
            print(
                "Error: PyQt5 GUI module not available. "
                "Install PyQt5 or run without --gui flag.",
                file=sys.stderr,
            )
            sys.exit(1)

        from PyQt5.QtWidgets import QApplication

        app = QApplication(sys.argv)
        window = SeniorMainWindow(pipeline=pipeline)
        window.show()
        sys.exit(app.exec_())

    def run(self, args=None) -> None:
        """Main entry point for the application.

        Parses arguments, builds the pipeline, and launches the
        appropriate mode (GUI or CLI).

        Args:
            args: Optional list of argument strings (for testing).
        """
        parsed_args = self.parse_args(args)

        if parsed_args.gui:
            pipeline = self._build_pipeline(parsed_args)
            self._run_gui(pipeline)
        else:
            pipeline = self._build_pipeline(parsed_args)
            self._run_cli(pipeline)


def main():
    """Console script entry point."""
    app = SeniorCitizenApp()
    app.run()


if __name__ == "__main__":
    main()
