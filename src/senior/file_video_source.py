"""File-based video source for the Senior Citizen Identification system.

Wraps OpenCV VideoCapture for reading frames from pre-recorded video files.
Supports MP4, AVI, and MOV formats.
"""

import os
from typing import Optional

import cv2
import numpy as np

from src.senior.exceptions import UnsupportedFormatError
from src.senior.video_source import VideoSource

# Supported video file extensions (case-insensitive)
SUPPORTED_FORMATS = {".mp4", ".avi", ".mov"}


class FileVideoSource(VideoSource):
    """Video source backed by a pre-recorded video file.

    Wraps cv2.VideoCapture(path) to provide frame-by-frame reading
    from MP4, AVI, or MOV video files.

    Attributes:
        path: Path to the video file.
    """

    def __init__(self, path: str) -> None:
        """Initialize FileVideoSource with the given file path.

        Args:
            path: Path to the video file to read.
        """
        self._path = path
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def path(self) -> str:
        """The file path for this video source."""
        return self._path

    def open(self) -> bool:
        """Open the video file for reading.

        Validates that the file exists and has a supported format
        (MP4, AVI, MOV) before attempting to open.

        Returns:
            True if the file was opened successfully, False otherwise.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedFormatError: If the file extension is not
                one of .mp4, .avi, or .mov (case-insensitive).
        """
        # Validate file existence
        if not os.path.isfile(self._path):
            raise FileNotFoundError(
                f"Video file not found: {self._path}"
            )

        # Validate format
        _, ext = os.path.splitext(self._path)
        if ext.lower() not in SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported video format '{ext}'. "
                f"Accepted formats: MP4, AVI, MOV."
            )

        # Open the video capture
        self._cap = cv2.VideoCapture(self._path)
        return self._cap.isOpened()

    def read_frame(self) -> Optional[np.ndarray]:
        """Read the next frame from the video file.

        Returns:
            The frame as a numpy ndarray (BGR format), or None if no frame
            is available (end of file, read error, or source not opened).
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        return frame

    def release(self) -> None:
        """Release the video capture and free associated resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        """Check whether the video source is currently opened.

        Returns:
            True if the source is open and ready for reading, False otherwise.
        """
        if self._cap is None:
            return False
        return self._cap.isOpened()

    def get_fps(self) -> float:
        """Get the frame rate of the video file.

        Returns:
            The frames per second (FPS) of the video file, or 0.0 if
            the source is not opened.
        """
        if self._cap is None or not self._cap.isOpened():
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)
