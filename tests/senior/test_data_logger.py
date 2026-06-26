"""Unit tests for the DataLogger class.

Tests cover:
1. CSV file creation with header
2. Records appended without overwriting
3. Default filename pattern matches detections_YYYYMMDD_HHMMSS.csv
4. Flush writes buffered records to disk
"""

import os
import re
import tempfile

import pandas as pd
import pytest

from src.senior.data_logger import DataLogger
from src.senior.models import DetectionRecord


@pytest.fixture
def tmp_csv(tmp_path):
    """Return a temporary CSV file path."""
    return str(tmp_path / "test_output.csv")


@pytest.fixture
def sample_record():
    """A sample DetectionRecord for testing."""
    return DetectionRecord(
        timestamp="2024-01-15T14:30:22",
        age=72,
        gender="Female",
        is_senior_citizen="Yes",
    )


@pytest.fixture
def sample_record_2():
    """A second sample DetectionRecord for testing appends."""
    return DetectionRecord(
        timestamp="2024-01-15T14:30:23",
        age=45,
        gender="Male",
        is_senior_citizen="No",
    )


class TestCSVFileCreationWithHeader:
    """Test that DataLogger creates a CSV file with proper header on first write."""

    def test_creates_csv_with_header_on_first_flush(self, tmp_csv, sample_record):
        """CSV file is created with header row on first flush."""
        logger = DataLogger(output_path=tmp_csv, format="csv")
        logger.log(sample_record)
        logger.flush()

        assert os.path.exists(tmp_csv)
        df = pd.read_csv(tmp_csv)
        assert list(df.columns) == ["Timestamp", "Age", "Gender", "Is_Senior_Citizen"]

    def test_first_record_written_correctly(self, tmp_csv, sample_record):
        """First record values are written correctly to CSV."""
        logger = DataLogger(output_path=tmp_csv, format="csv")
        logger.log(sample_record)
        logger.flush()

        df = pd.read_csv(tmp_csv)
        assert len(df) == 1
        assert df.iloc[0]["Timestamp"] == "2024-01-15T14:30:22"
        assert df.iloc[0]["Age"] == 72
        assert df.iloc[0]["Gender"] == "Female"
        assert df.iloc[0]["Is_Senior_Citizen"] == "Yes"

    def test_file_not_created_before_flush(self, tmp_csv, sample_record):
        """CSV file is not created until flush is called."""
        logger = DataLogger(output_path=tmp_csv, format="csv")
        logger.log(sample_record)
        # Don't flush yet
        assert not os.path.exists(tmp_csv)


class TestAppendWithoutOverwriting:
    """Test that subsequent writes append records without overwriting."""

    def test_second_flush_appends_records(self, tmp_csv, sample_record, sample_record_2):
        """Second flush appends new records without overwriting first ones."""
        logger = DataLogger(output_path=tmp_csv, format="csv")

        # First write
        logger.log(sample_record)
        logger.flush()

        # Second write
        logger.log(sample_record_2)
        logger.flush()

        df = pd.read_csv(tmp_csv)
        assert len(df) == 2
        assert df.iloc[0]["Age"] == 72
        assert df.iloc[1]["Age"] == 45

    def test_multiple_records_in_single_flush(self, tmp_csv, sample_record, sample_record_2):
        """Multiple records buffered before flush are all written."""
        logger = DataLogger(output_path=tmp_csv, format="csv", flush_interval=999)

        logger.log(sample_record)
        logger.log(sample_record_2)
        logger.flush()

        df = pd.read_csv(tmp_csv)
        assert len(df) == 2

    def test_original_records_preserved_after_append(self, tmp_csv, sample_record, sample_record_2):
        """Original records remain unchanged after appending new ones."""
        logger = DataLogger(output_path=tmp_csv, format="csv")

        logger.log(sample_record)
        logger.flush()

        # Read state after first write
        df_first = pd.read_csv(tmp_csv)

        logger.log(sample_record_2)
        logger.flush()

        # Verify first record is still intact
        df_second = pd.read_csv(tmp_csv)
        assert df_second.iloc[0]["Timestamp"] == df_first.iloc[0]["Timestamp"]
        assert df_second.iloc[0]["Age"] == df_first.iloc[0]["Age"]
        assert df_second.iloc[0]["Gender"] == df_first.iloc[0]["Gender"]


class TestDefaultFilenamePattern:
    """Test that default filename matches detections_YYYYMMDD_HHMMSS.csv pattern."""

    def test_default_filename_matches_pattern(self):
        """Generated filename matches detections_YYYYMMDD_HHMMSS.csv."""
        logger = DataLogger(output_path=None, format="csv")
        filename = os.path.basename(logger.output_path)

        pattern = r"^detections_\d{8}_\d{6}\.csv$"
        assert re.match(pattern, filename), (
            f"Filename '{filename}' does not match pattern detections_YYYYMMDD_HHMMSS.csv"
        )

    def test_default_filename_is_in_cwd(self):
        """Default file is created in the current working directory."""
        logger = DataLogger(output_path=None, format="csv")
        expected_dir = os.getcwd()
        actual_dir = os.path.dirname(logger.output_path)
        assert actual_dir == expected_dir

    def test_default_filename_contains_valid_datetime(self):
        """The timestamp portion of the default filename represents a valid datetime."""
        logger = DataLogger(output_path=None, format="csv")
        filename = os.path.basename(logger.output_path)

        # Extract date and time portions
        match = re.match(r"^detections_(\d{8})_(\d{6})\.csv$", filename)
        assert match is not None

        date_str = match.group(1)
        time_str = match.group(2)

        # Verify date is parseable
        from datetime import datetime
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        assert dt is not None


class TestFlushWritesBufferedRecords:
    """Test that flush() writes all buffered records to disk."""

    def test_flush_writes_buffer_to_disk(self, tmp_csv, sample_record):
        """Calling flush writes buffered records to the file."""
        logger = DataLogger(output_path=tmp_csv, format="csv", flush_interval=999)

        logger.log(sample_record)
        # Buffer should have 1 record, file should not exist
        assert not os.path.exists(tmp_csv)

        logger.flush()
        # Now file should exist with the record
        assert os.path.exists(tmp_csv)
        df = pd.read_csv(tmp_csv)
        assert len(df) == 1

    def test_flush_clears_buffer(self, tmp_csv, sample_record, sample_record_2):
        """After flush, buffer is empty — flushing again doesn't duplicate."""
        logger = DataLogger(output_path=tmp_csv, format="csv", flush_interval=999)

        logger.log(sample_record)
        logger.flush()

        # Flush again without new records
        logger.flush()

        df = pd.read_csv(tmp_csv)
        assert len(df) == 1  # No duplicates

    def test_close_flushes_remaining_buffer(self, tmp_csv, sample_record):
        """Calling close() flushes any remaining buffered records."""
        logger = DataLogger(output_path=tmp_csv, format="csv", flush_interval=999)

        logger.log(sample_record)
        logger.close()

        assert os.path.exists(tmp_csv)
        df = pd.read_csv(tmp_csv)
        assert len(df) == 1

    def test_auto_flush_on_interval(self, tmp_csv, sample_record):
        """Records are auto-flushed when flush_interval has elapsed."""
        logger = DataLogger(output_path=tmp_csv, format="csv", flush_interval=0)
        # flush_interval=0 means flush on every log call
        logger.log(sample_record)

        assert os.path.exists(tmp_csv)
        df = pd.read_csv(tmp_csv)
        assert len(df) == 1
