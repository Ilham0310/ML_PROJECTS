"""
Pytest configuration to prevent module mocking from polluting the test environment.

Some tests mock heavy dependencies (tensorflow, cv2) at the sys.modules level.
This conftest ensures real modules are imported first, so the preprocessor tests
that depend on real cv2/numpy behavior work correctly regardless of test order.
"""

import sys
import importlib.machinery

# Ensure real cv2 is imported before any test can mock it
# This prevents test_model_load.py from polluting sys.modules['cv2'] for later tests
import cv2  # noqa: F401
import numpy  # noqa: F401


def _ensure_tensorflow_mock_specs():
    """Give mocked tensorflow modules enough import metadata for find_spec()."""

    for mod_name, module in list(sys.modules.items()):
        if mod_name == "tensorflow" or mod_name.startswith("tensorflow."):
            if getattr(module, "__spec__", None) is None:
                module.__spec__ = importlib.machinery.ModuleSpec(
                    mod_name,
                    loader=None,
                )


def pytest_collect_file(file_path, parent):  # noqa: D401
    """Prevent TensorFlow sys.modules mocks from breaking later collection."""

    _ensure_tensorflow_mock_specs()
    return None
