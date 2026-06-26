"""Unit tests for VideoSource classes.

Tests file format validation, file existence checking, and webcam
index validation logic for the Senior Citizen Identification system.
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from src.senior.file_video_source import FileVideoSource, SUPPORTED_FORMATS
from src.senior.webcam_video_source import WebcamVideoSource
from src.senior.exceptions import UnsupportedFormatError, CameraUnavailableError


class TestFileVideoSourceFormatValidation:
    """Tests for FileVideoSource file format validation logic."""

    def test_supported_formats_are_mp4_avi_mov(self):
        """SUPPORTED_FORMATS contains exactly the expected extensions."""
        assert SUPPORTED_FORMATS == {".mp4", ".avi", ".mov"}

    @pytest.mark.parametrize("ext", [".mp4", ".avi", ".mov"])
    def test_open_accepts_supported_format(self, ext, tmp_path):
        """open() does not raise UnsupportedFormatError for supported formats."""
        # Create a dummy file with the supported extension
        video_file = tmp_path / f"test_video{ext}"
        video_file.write_bytes(b"\x00" * 100)

        source = FileVideoSource(str(video_file))
        # open() should not raise UnsupportedFormatError
        # It may return False because the file isn't a real video,
        # but the format validation itself should pass.
        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_cap.return_value = mock_instance
            result = source.open()
            assert result is True

    @pytest.mark.parametrize("ext", [".mp4", ".AVI", ".Mov", ".MP4"])
    def test_open_accepts_case_insensitive_formats(self, ext, tmp_path):
        """open() validates format case-insensitively."""
        video_file = tmp_path / f"test_video{ext}"
        video_file.write_bytes(b"\x00" * 100)

        source = FileVideoSource(str(video_file))
        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_cap.return_value = mock_instance
            result = source.open()
            assert result is True

    @pytest.mark.parametrize("ext", [".mkv", ".wmv", ".flv", ".webm", ".txt", ".jpg"])
    def test_open_rejects_unsupported_format(self, ext, tmp_path):
        """open() raises UnsupportedFormatError for unsupported extensions."""
        video_file = tmp_path / f"test_video{ext}"
        video_file.write_bytes(b"\x00" * 100)

        source = FileVideoSource(str(video_file))
        with pytest.raises(UnsupportedFormatError) as exc_info:
            source.open()
        assert "Unsupported video format" in str(exc_info.value)
        assert "MP4, AVI, MOV" in str(exc_info.value)

    def test_open_raises_file_not_found_for_missing_file(self):
        """open() raises FileNotFoundError when file does not exist."""
        source = FileVideoSource("/nonexistent/path/video.mp4")
        with pytest.raises(FileNotFoundError) as exc_info:
            source.open()
        assert "Video file not found" in str(exc_info.value)

    def test_open_checks_existence_before_format(self, tmp_path):
        """open() raises FileNotFoundError before checking format for missing files."""
        source = FileVideoSource("/nonexistent/path/video.mkv")
        with pytest.raises(FileNotFoundError):
            source.open()

    def test_read_frame_returns_none_when_not_opened(self):
        """read_frame() returns None if source was never opened."""
        source = FileVideoSource("dummy.mp4")
        assert source.read_frame() is None

    def test_is_opened_returns_false_initially(self):
        """is_opened() returns False before open() is called."""
        source = FileVideoSource("dummy.mp4")
        assert source.is_opened() is False

    def test_get_fps_returns_zero_when_not_opened(self):
        """get_fps() returns 0.0 when source is not opened."""
        source = FileVideoSource("dummy.mp4")
        assert source.get_fps() == 0.0

    def test_release_is_safe_when_not_opened(self):
        """release() does not raise when called before open()."""
        source = FileVideoSource("dummy.mp4")
        source.release()  # Should not raise

    def test_read_frame_returns_none_after_release(self, tmp_path):
        """read_frame() returns None after release() is called."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00" * 100)

        source = FileVideoSource(str(video_file))
        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_cap.return_value = mock_instance
            source.open()

        source.release()
        assert source.read_frame() is None


class TestWebcamVideoSourceValidation:
    """Tests for WebcamVideoSource index validation and timeout logic."""

    def test_valid_camera_index_range(self):
        """Constructor accepts indices 0 through 10."""
        for i in range(11):
            source = WebcamVideoSource(camera_index=i)
            assert source.camera_index == i

    def test_negative_camera_index_raises_value_error(self):
        """Constructor raises ValueError for negative index."""
        with pytest.raises(ValueError) as exc_info:
            WebcamVideoSource(camera_index=-1)
        assert "must be between 0 and 10" in str(exc_info.value)

    def test_camera_index_above_10_raises_value_error(self):
        """Constructor raises ValueError for index > 10."""
        with pytest.raises(ValueError) as exc_info:
            WebcamVideoSource(camera_index=11)
        assert "must be between 0 and 10" in str(exc_info.value)

    def test_default_timeout_is_5_seconds(self):
        """Default timeout_sec is 5.0."""
        source = WebcamVideoSource(camera_index=0)
        assert source.timeout_sec == 5.0

    def test_open_raises_camera_unavailable_on_timeout(self):
        """open() raises CameraUnavailableError when camera doesn't connect in time."""
        source = WebcamVideoSource(camera_index=0, timeout_sec=0.3)

        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = False
            mock_cap.return_value = mock_instance

            with pytest.raises(CameraUnavailableError) as exc_info:
                source.open()
            assert "could not be opened" in str(exc_info.value)

    def test_is_opened_returns_false_initially(self):
        """is_opened() returns False before open() is called."""
        source = WebcamVideoSource(camera_index=0)
        assert source.is_opened() is False

    def test_get_fps_returns_zero_when_not_opened(self):
        """get_fps() returns 0.0 when webcam is not opened."""
        source = WebcamVideoSource(camera_index=0)
        assert source.get_fps() == 0.0

    def test_read_frame_returns_none_when_not_opened(self):
        """read_frame() returns None if camera is not opened."""
        source = WebcamVideoSource(camera_index=0)
        assert source.read_frame() is None
