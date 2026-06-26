"""
Tests for the Preprocessor module.
Validates Requirements 7.1, 7.2, 7.3, 7.4
"""

import numpy as np
import cv2
import os
import tempfile
import pytest

from src.data.preprocessor import Preprocessor


@pytest.fixture
def preprocessor():
    return Preprocessor()


@pytest.fixture
def sample_image_path():
    """Use a real image from the dataset for testing."""
    path = os.path.join("data", "crop_part1", "1_0_0_20161219140623097.jpg.chip.jpg")
    if os.path.exists(path):
        return path
    pytest.skip("Sample image not available")


@pytest.fixture
def temp_image_path():
    """Create a temporary test image."""
    img = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, img)
        yield f.name
    os.unlink(f.name)


class TestPreprocessResize:
    """Requirement 7.1: Resize to 224x224"""

    def test_output_shape_is_224x224x3(self, preprocessor, temp_image_path):
        result = preprocessor.preprocess(temp_image_path)
        assert result.shape == (224, 224, 3)

    def test_resize_from_different_dimensions(self, preprocessor):
        """Test that images of various sizes are resized to 224x224."""
        for size in [(50, 50), (300, 400), (1024, 768), (10, 500)]:
            img = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cv2.imwrite(path, img)
            result = preprocessor.preprocess(path)
            assert result.shape == (224, 224, 3)
            os.unlink(path)


class TestPreprocessNormalization:
    """Requirement 7.2: Normalize to [-1, 1] using MobileNetV2 preprocess_input"""

    def test_output_values_in_minus1_to_1(self, preprocessor, temp_image_path):
        result = preprocessor.preprocess(temp_image_path)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_output_dtype_is_float(self, preprocessor, temp_image_path):
        result = preprocessor.preprocess(temp_image_path)
        assert result.dtype == np.float32


class TestPreprocessBGRToRGB:
    """Requirement 7.3: BGR to RGB conversion"""

    def test_bgr_to_rgb_conversion(self, preprocessor):
        """Create an image with known BGR values and verify conversion."""
        # Create an image that is pure blue in BGR: B=255, G=0, R=0
        # After BGR->RGB conversion it becomes: R=0, G=0, B=255
        # After MobileNetV2 preprocess_input (x/127.5 - 1): 0 -> -1, 255 -> 1
        # So channel 0 (R) should be -1, channel 2 (B) should be 1
        blue_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        blue_bgr[:, :, 0] = 255  # Blue channel in BGR

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, blue_bgr)
        result = preprocessor.preprocess(path)
        # After BGR->RGB: R=0 -> -1, B=255 -> 1
        # Channel 2 (Blue in RGB) should be the highest
        assert result[:, :, 2].mean() > result[:, :, 0].mean()
        os.unlink(path)


class TestAugmentation:
    """Requirement 7.4: Augmentation with flip, rotation, brightness"""

    def test_augment_preserves_shape(self, preprocessor):
        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        result = preprocessor.augment(image)
        assert result.shape == (224, 224, 3)

    def test_augment_output_in_valid_range(self, preprocessor):
        """Augmented output should remain in [-1, 1]."""
        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        # Run multiple times to cover random augmentations
        for _ in range(10):
            result = preprocessor.augment(image)
            assert result.min() >= -1.0 - 0.01  # small tolerance for float precision
            assert result.max() <= 1.0 + 0.01

    def test_augment_rejects_wrong_shape(self, preprocessor):
        image = np.random.uniform(-1, 1, (100, 100, 3)).astype(np.float32)
        with pytest.raises(ValueError):
            preprocessor.augment(image)

    def test_augment_produces_variation(self, preprocessor):
        """Running augment multiple times should produce different results (randomness)."""
        image = np.random.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
        results = [preprocessor.augment(image) for _ in range(20)]
        # At least some results should differ (flip or rotation changes pixels)
        differences = [not np.array_equal(results[0], r) for r in results[1:]]
        assert any(differences), "Augmentation should introduce variation"


class TestValidateImageFile:
    """Requirement 7.3 (extensions) + file readability"""

    def test_valid_jpg_file(self, temp_image_path):
        is_valid, error = Preprocessor.validate_image_file(temp_image_path)
        assert is_valid is True
        assert error is None

    def test_valid_png_file(self):
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, img)
        is_valid, error = Preprocessor.validate_image_file(path)
        assert is_valid is True
        os.unlink(path)

    def test_valid_bmp_file(self):
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        fd, path = tempfile.mkstemp(suffix=".bmp")
        os.close(fd)
        cv2.imwrite(path, img)
        is_valid, error = Preprocessor.validate_image_file(path)
        assert is_valid is True
        os.unlink(path)

    def test_unsupported_extension(self):
        fd, path = tempfile.mkstemp(suffix=".gif")
        os.write(fd, b"fake content")
        os.close(fd)
        is_valid, error = Preprocessor.validate_image_file(path)
        assert is_valid is False
        assert "Unsupported" in error
        os.unlink(path)

    def test_nonexistent_file(self):
        is_valid, error = Preprocessor.validate_image_file("/nonexistent/path/image.jpg")
        assert is_valid is False
        assert "does not exist" in error

    def test_corrupted_file(self):
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.write(fd, b"this is not an image")
        os.close(fd)
        is_valid, error = Preprocessor.validate_image_file(path)
        assert is_valid is False
        assert "corrupted" in error.lower() or "could not read" in error.lower()
        os.unlink(path)
