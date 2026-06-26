"""Shared exceptions for optional feature modules."""


class ModelLoadError(Exception):
    """Raised when model weights are absent or cannot be loaded."""


class ClassificationError(Exception):
    """Raised when a classifier cannot produce a valid prediction."""
