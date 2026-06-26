"""Webcam video source for the Senior Citizen Identification system.

Provides a concrete VideoSource implementation that captures frames
from a live webcam device via OpenCV's VideoCapture.
"""

import time
from typing import Optional

import cv2
import numpy as np

from src.senior.exceptions import CameraUnavailableError
from src.senior.video_source import VideoSource


class WebcamVideoSource(VideoSource):
    """Video source that captures frames from a live webcam.

    Wraps cv2.VideoCapture with a camera device index and implements
    a 5-second connection timeout during open().

    Attributes:
        camera_index: Integer index of the camera device (0–10).
        timeout_sec: Maximum seconds to wait for camera connection.
    """

    def __init__(self, camera_index: int = 0, timeout_sec: float = 5.0) -> None:
        """Initialize the webcam video source.

        Args:
            camera_index: Camera device index in the range 0–10.
            timeout_sec: Maximum time in seconds to wait for camera
                connection during open(). Defaults to 5.0.

        Raises:
            ValueError: If camera_index is not in the range 0–10.
        """
        if not (0 <= camera_index <= 10):
            raise ValueError(
                f"camera_index must be between 0 and 10, got {camera_index}"
            )
        self._camera_index = camera_index
        self._timeout_sec = timeout_sec
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def camera_index(self) -> int:
        """The camera device index."""
        return self._camera_index

    @property
    def timeout_sec(self) -> float:
        """The connection timeout in seconds."""
        return self._timeout_sec

    def open(self) -> bool:
        """Open the webcam device for frame capture.

        Attempts to connect to the camera within the configured timeout
        period. Polls the device at short intervals until either a
        successful connection is made or the timeout expires.

        Returns:
            True if the camera was opened successfully.

        Raises:
            CameraUnavailableError: If the camera cannot be opened
                within the timeout period.
        """
        self._cap = cv2.VideoCapture(self._camera_index)

        start_time = time.monotonic()
        while not self._cap.isOpened():
            elapsed = time.monotonic() - start_time
            if elapsed >= self._timeout_sec:
                self._cap.release()
                self._cap = None
                raise CameraUnavailableError(
                    f"Camera at index {self._camera_index} could not be "
                    f"opened within {self._timeout_sec} seconds."
                )
            time.sleep(0.1)
            self._cap.open(self._camera_index)

        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Read the next frame from the webcam.

        Returns:
            The frame as a numpy ndarray in BGR format, or None if the
            camera is not opened or the read fails.
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        """Release the webcam device and free resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        """Check whether the webcam is currently opened.

        Returns:
            True if the camera is open and ready for reading.
        """
        return self._cap is not None and self._cap.isOpened()

    def get_fps(self) -> float:
        """Get the frame rate of the webcam.

        Returns:
            The camera's native capture rate in frames per second.
            Returns 0.0 if the camera is not opened.
        """
        if self._cap is None or not self._cap.isOpened():
            return 0.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0
