"""Unit tests for the SeniorCitizenApp entry point.

Tests verify argument parsing, particularly the --gui flag behavior
and mode selection logic.
"""

import pytest

from src.senior.app import SeniorCitizenApp


class TestParseArgs:
    """Tests for SeniorCitizenApp.parse_args()."""

    def setup_method(self):
        self.app = SeniorCitizenApp()

    def test_gui_flag_present_sets_gui_true(self):
        """--gui flag present sets gui=True."""
        args = self.app.parse_args(["--gui"])
        assert args.gui is True

    def test_gui_flag_absent_sets_gui_false(self):
        """Without --gui flag, gui defaults to False (CLI mode)."""
        args = self.app.parse_args([])
        assert args.gui is False

    def test_default_source_is_camera_index_zero(self):
        """Default source is camera index 0."""
        args = self.app.parse_args([])
        assert args.source == 0

    def test_source_file_path(self):
        """File path source is returned as string."""
        args = self.app.parse_args(["--source", "video.mp4"])
        assert args.source == "video.mp4"
        assert isinstance(args.source, str)

    def test_source_camera_index(self):
        """Camera index source is returned as int."""
        args = self.app.parse_args(["--source", "2"])
        assert args.source == 2
        assert isinstance(args.source, int)

    def test_default_format_is_csv(self):
        """Default output format is csv."""
        args = self.app.parse_args([])
        assert args.format == "csv"

    def test_format_excel(self):
        """--format excel sets format to excel."""
        args = self.app.parse_args(["--format", "excel"])
        assert args.format == "excel"

    def test_default_output_is_none(self):
        """Default output path is None (auto-generated)."""
        args = self.app.parse_args([])
        assert args.output is None

    def test_output_path_specified(self):
        """Custom output path is correctly parsed."""
        args = self.app.parse_args(["--output", "results.csv"])
        assert args.output == "results.csv"

    def test_default_model_dir(self):
        """Default model directory is 'models'."""
        args = self.app.parse_args([])
        assert args.model_dir == "models"

    def test_custom_model_dir(self):
        """Custom model directory is correctly parsed."""
        args = self.app.parse_args(["--model-dir", "/path/to/models"])
        assert args.model_dir == "/path/to/models"

    def test_gui_with_source_and_output(self):
        """Multiple arguments can be combined with --gui flag."""
        args = self.app.parse_args([
            "--gui",
            "--source", "test.avi",
            "--output", "output.xlsx",
            "--format", "excel",
            "--model-dir", "custom_models",
        ])
        assert args.gui is True
        assert args.source == "test.avi"
        assert args.output == "output.xlsx"
        assert args.format == "excel"
        assert args.model_dir == "custom_models"

    def test_invalid_format_raises_error(self):
        """Invalid format value causes SystemExit."""
        with pytest.raises(SystemExit):
            self.app.parse_args(["--format", "json"])

    def test_invalid_camera_index_out_of_range(self):
        """Camera index > 10 causes SystemExit."""
        with pytest.raises(SystemExit):
            self.app.parse_args(["--source", "15"])

    def test_source_mov_file(self):
        """MOV files are recognized as file sources."""
        args = self.app.parse_args(["--source", "recording.mov"])
        assert args.source == "recording.mov"
        assert isinstance(args.source, str)
