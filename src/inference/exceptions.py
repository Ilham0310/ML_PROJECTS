"""Compatibility wrapper for Nationality Detection inference exceptions."""

from src.nationality.exceptions import (
    CORRUPT_IMAGE_MESSAGE,
    FILE_SIZE_MESSAGE,
    GENERIC_INFERENCE_MESSAGE,
    INVALID_FORMAT_MESSAGE,
    MODEL_LOAD_MESSAGE,
    MULTIPLE_FACES_MESSAGE,
    NO_FACE_MESSAGE,
    CorruptImageError,
    FileSizeError,
    InferenceError,
    InferenceTimeoutError,
    InvalidFileFormatError,
    ModelLoadError,
    MultipleFacesError,
    NoFaceDetectedError,
)

__all__ = [
    "CORRUPT_IMAGE_MESSAGE",
    "FILE_SIZE_MESSAGE",
    "GENERIC_INFERENCE_MESSAGE",
    "INVALID_FORMAT_MESSAGE",
    "MODEL_LOAD_MESSAGE",
    "MULTIPLE_FACES_MESSAGE",
    "NO_FACE_MESSAGE",
    "CorruptImageError",
    "FileSizeError",
    "InferenceError",
    "InferenceTimeoutError",
    "InvalidFileFormatError",
    "ModelLoadError",
    "MultipleFacesError",
    "NoFaceDetectedError",
]
