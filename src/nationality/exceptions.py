"""Exception hierarchy for Nationality Detection."""


class InferenceError(Exception):
    """Base exception for nationality inference errors."""


class InvalidFileFormatError(InferenceError):
    """Raised when an uploaded file extension is not supported."""


class FileSizeError(InferenceError):
    """Raised when an uploaded file is larger than 10 MB."""


class CorruptImageError(InferenceError):
    """Raised when OpenCV cannot decode the image."""


class NoFaceDetectedError(InferenceError):
    """Raised when no face is detected."""


class MultipleFacesError(InferenceError):
    """Raised when more than one face is detected."""


class ModelLoadError(InferenceError):
    """Raised when model weights are absent or corrupted."""


class InferenceTimeoutError(InferenceError):
    """Raised when inference exceeds the configured timeout."""


INVALID_FORMAT_MESSAGE = (
    "The file format is not supported. Please upload a JPEG, PNG, or BMP image."
)
FILE_SIZE_MESSAGE = "File exceeds the 10 MB size limit. Please choose a smaller image."
CORRUPT_IMAGE_MESSAGE = (
    "The selected file could not be opened. Please choose a valid image."
)
NO_FACE_MESSAGE = (
    "No face was detected in the image. Please upload an image with a clearly visible face."
)
MULTIPLE_FACES_MESSAGE = (
    "Multiple faces detected. Please upload an image containing a single face."
)
MODEL_LOAD_MESSAGE = (
    "Model files are missing or corrupted. Please run `train.py` before launching the application."
)
GENERIC_INFERENCE_MESSAGE = (
    "An error occurred during prediction. Please try again with a different image."
)
