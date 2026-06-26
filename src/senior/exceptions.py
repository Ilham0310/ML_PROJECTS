"""Custom exceptions for the Senior Citizen Identification system.

Provides a hierarchy of domain-specific exceptions for error handling
across video input, model inference, and data logging components.
"""


class SeniorIdError(Exception):
    """Base exception for the Senior Citizen Identification system."""

    pass


class VideoSourceError(SeniorIdError):
    """Base exception for video source related errors."""

    pass


class UnsupportedFormatError(VideoSourceError):
    """Raised when a video file has an unsupported format.

    Supported formats are MP4, AVI, and MOV.
    """

    pass


class CameraUnavailableError(VideoSourceError):
    """Raised when a webcam device cannot be opened within the timeout period."""

    pass


class FeedInterruptionError(VideoSourceError):
    """Raised when the video feed experiences prolonged interruption.

    Triggered after 50+ consecutive frame read failures and unsuccessful
    reconnection attempts over 30 seconds.
    """

    pass


class ModelPredictionError(SeniorIdError):
    """Raised when a model fails to produce a prediction for a detected person."""

    pass


class DataLoggerError(SeniorIdError):
    """Raised when the data logger encounters a write or flush error."""

    pass
