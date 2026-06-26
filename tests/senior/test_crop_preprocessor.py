"""Unit tests for CropPreprocessor class.

Tests verify output shape, value range, boundary clipping, and error handling.
"""

import numpy as np
import pytest

from src.senior.crop_preprocessor import CropPreprocessor
from src.senior.models import BoundingBox


@pytest.fixture
def preprocessor() -> CropPreprocessor:
    """Create a CropPreprocessor instance."""
    return CropPreprocessor()


@pytest.fixture
def color_frame() -> np.ndarray:
    """A 480x640 BGR frame with random pixel values."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


class TestCropPreprocessorOutputShape:
    """Tests verifying the output shape is always (224, 224, 3)."""

    def test_output_shape_standard_bbox(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """Output shape is (224, 224, 3) for a standard bounding box."""
        bbox = BoundingBox(x=100, y=50, width=64, height=128)
        result = preprocessor.crop_and_preprocess(color_frame, bbox)
        assert result.shape == (224, 224, 3)

    def test_output_shape_large_bbox(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """Output shape is (224, 224, 3) for a large bounding box."""
        bbox = BoundingBox(x=0, y=0, width=400, height=400)
        result = preprocessor.crop_and_preprocess(color_frame, bbox)
        assert result.shape == (224, 224, 3)

    def test_output_shape_small_bbox(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """Output shape is (224, 224, 3) for a minimal 1x1 bounding box."""
        bbox = BoundingBox(x=50, y=50, width=1, height=1)
        result = preprocessor.crop_and_preprocess(color_frame, bbox)
        assert result.shape == (224, 224, 3)

    def test_output_dtype_is_float32(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """Output dtype is float32."""
        bbox = BoundingBox(x=100, y=50, width=64, height=128)
        result = preprocessor.crop_and_preprocess(color_frame, bbox)
        assert result.dtype == np.float32


class TestCropPreprocessorValueRange:
    """Tests verifying output values are normalized to [-1, 1]."""

    def test_values_in_range_negative_one_to_one(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """All output values are in [-1, 1] for a random color frame."""
        bbox = BoundingBox(x=100, y=50, width=200, height=200)
        result = preprocessor.crop_and_preprocess(color_frame, bbox)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_black_frame_produces_minus_one(self, preprocessor: CropPreprocessor):
        """A black frame (all zeros) normalizes to -1.0."""
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        result = preprocessor.crop_and_preprocess(black_frame, bbox)
        assert np.allclose(result, -1.0)

    def test_white_frame_produces_one(self, preprocessor: CropPreprocessor):
        """A white frame (all 255) normalizes to 1.0."""
        white_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        result = preprocessor.crop_and_preprocess(white_frame, bbox)
        assert np.allclose(result, 1.0)

    def test_mid_gray_produces_approximately_zero(self, preprocessor: CropPreprocessor):
        """A mid-gray frame (value ~127-128) normalizes to approximately 0."""
        gray_frame = np.full((480, 640, 3), 127, dtype=np.uint8)
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        result = preprocessor.crop_and_preprocess(gray_frame, bbox)
        # 127 / 127.5 - 1.0 ≈ -0.00392
        assert np.all(np.abs(result) < 0.01)


class TestCropPreprocessorBoundaryClipping:
    """Tests verifying bbox clipping at frame boundaries."""

    def test_bbox_partially_outside_right_edge(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """Bbox extending beyond right edge is clipped correctly."""
        # Frame is 640 wide, bbox extends to x=600+100=700
        bbox = BoundingBox(x=600, y=50, width=100, height=100)
        result = preprocessor.crop_and_preprocess(color_frame, bbox)
        assert result.shape == (224, 224, 3)

    def test_bbox_partially_outside_bottom_edge(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """Bbox extending beyond bottom edge is clipped correctly."""
        # Frame is 480 tall, bbox extends to y=450+100=550
        bbox = BoundingBox(x=50, y=450, width=100, height=100)
        result = preprocessor.crop_and_preprocess(color_frame, bbox)
        assert result.shape == (224, 224, 3)

    def test_bbox_with_negative_x(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """Bbox with negative x coordinate is clipped to frame left edge."""
        bbox = BoundingBox(x=-20, y=50, width=100, height=100)
        result = preprocessor.crop_and_preprocess(color_frame, bbox)
        assert result.shape == (224, 224, 3)

    def test_bbox_with_negative_y(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """Bbox with negative y coordinate is clipped to frame top edge."""
        bbox = BoundingBox(x=50, y=-30, width=100, height=100)
        result = preprocessor.crop_and_preprocess(color_frame, bbox)
        assert result.shape == (224, 224, 3)


class TestCropPreprocessorErrors:
    """Tests verifying error handling for invalid inputs."""

    def test_zero_width_raises_value_error(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """Zero-width bbox raises ValueError."""
        bbox = BoundingBox(x=100, y=50, width=0, height=100)
        with pytest.raises(ValueError, match="zero-area crop"):
            preprocessor.crop_and_preprocess(color_frame, bbox)

    def test_zero_height_raises_value_error(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """Zero-height bbox raises ValueError."""
        bbox = BoundingBox(x=100, y=50, width=100, height=0)
        with pytest.raises(ValueError, match="zero-area crop"):
            preprocessor.crop_and_preprocess(color_frame, bbox)

    def test_bbox_completely_outside_frame_raises_value_error(self, preprocessor: CropPreprocessor, color_frame: np.ndarray):
        """Bbox entirely outside frame boundaries raises ValueError."""
        bbox = BoundingBox(x=700, y=500, width=50, height=50)
        with pytest.raises(ValueError, match="zero-area crop"):
            preprocessor.crop_and_preprocess(color_frame, bbox)


class TestCropPreprocessorTargetSize:
    """Tests verifying the TARGET_SIZE class attribute."""

    def test_target_size_is_224x224(self):
        """TARGET_SIZE constant is (224, 224)."""
        assert CropPreprocessor.TARGET_SIZE == (224, 224)
