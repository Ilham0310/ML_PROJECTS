"""Abstract video source interface for the Senior Citizen Identification system.

Provides a uniform interface for reading frames from video files or webcams.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class VideoSource(ABC):
    """Abstract base class for video input sources.

    Defines the interface for reading frames from either a pre-recorded
    video file or a live webcam feed.
    """

    @abstractmethod
    def open(self) -> bool:
        """Open the video source for reading.

        Returns:
            True if the source was opened successfully, False otherwise.
        """
        ...

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """Read the next frame from the video source.

        Returns:
            The frame as a numpy ndarray (BGR format), or None if no frame
            is available (end of file, read error, or source not opened).
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Release the video source and free associated resources."""
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        """Check whether the video source is currently opened.

        Returns:
            True if the source is open and ready for reading, False otherwise.
        """
        ...

    @abstractmethod
    def get_fps(self) -> float:
        """Get the frame rate of the video source.

        Returns:
            The frames per second (FPS) of the source. For webcams, this
            is the camera's native capture rate. For files, this is the
            video's encoded frame rate.
        """
        ...
