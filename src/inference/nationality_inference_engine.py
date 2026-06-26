"""Compatibility wrapper for the Nationality Inference Engine."""

from src.nationality.inference_engine import (
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
    NationalityInferenceEngine,
)

__all__ = ["MAX_FILE_SIZE_BYTES", "SUPPORTED_EXTENSIONS", "NationalityInferenceEngine"]
