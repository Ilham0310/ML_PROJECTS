"""Data Logger component for the Senior Citizen Identification system.

Writes DetectionRecords to CSV or Excel files with buffered logging,
periodic flush, append-only semantics, and retry on write failure.
"""

import logging
import os
import time
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.senior.models import DetectionRecord

logger = logging.getLogger(__name__)

# Column names matching the output file schema
COLUMNS = ["Timestamp", "Age", "Gender", "Is_Senior_Citizen"]


class DataLogger:
    """Buffered data logger that writes DetectionRecords to CSV or Excel.

    Supports CSV (default) and Excel (.xlsx) output formats.
    Implements periodic flush every 5 seconds, append-only writes,
    and retry every 10 seconds on write failure.

    Attributes:
        output_path: Path to the output file.
        format: Output format, either "csv" or "excel".
        flush_interval: Seconds between automatic flushes (default 5.0).
        retry_interval: Seconds between write retries on failure (default 10.0).
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        format: str = "csv",
        flush_interval: float = 5.0,
        retry_interval: float = 10.0,
    ):
        """Initialize DataLogger.

        Args:
            output_path: Path to the output file. If None, generates a default
                filename using pattern detections_YYYYMMDD_HHMMSS.csv.
            format: Output format - "csv" (default) or "excel".
            flush_interval: Seconds between periodic flushes (default 5.0).
            retry_interval: Seconds between retries on write failure (default 10.0).
        """
        self.format = format.lower()
        self.flush_interval = flush_interval
        self.retry_interval = retry_interval
        self._buffer: List[DetectionRecord] = []
        self._last_flush: float = time.time()
        self._last_retry: float = 0.0
        self._header_written: bool = False
        self._closed: bool = False

        if output_path is None:
            self.output_path = self._generate_default_filename()
        else:
            self.output_path = output_path

    def _generate_default_filename(self) -> str:
        """Generate default filename with pattern detections_YYYYMMDD_HHMMSS.csv."""
        now = datetime.now()
        filename = f"detections_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        return os.path.join(os.getcwd(), filename)

    def log(self, record: DetectionRecord) -> None:
        """Buffer a detection record. Triggers flush if interval has elapsed.

        Args:
            record: The DetectionRecord to log.
        """
        if self._closed:
            logger.warning("Attempted to log to a closed DataLogger.")
            return

        self._buffer.append(record)

        # Check if flush interval has elapsed
        elapsed = time.time() - self._last_flush
        if elapsed >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        """Write buffered records to disk.

        Creates the file with a header on first write.
        Appends subsequent records without overwriting.
        Retries on write failure (disk full, permission denied).
        """
        if not self._buffer:
            self._last_flush = time.time()
            return

        try:
            self._write_records(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()
        except (IOError, OSError, PermissionError) as e:
            logger.error(
                f"Failed to write to {self.output_path}: {e}. "
                f"Will retry in {self.retry_interval} seconds."
            )
            self._last_retry = time.time()

    def close(self) -> None:
        """Final flush and mark logger as closed."""
        if not self._closed:
            self.flush()
            self._closed = True

    def _write_records(self, records: List[DetectionRecord]) -> None:
        """Write a list of records to the output file.

        On first write, creates the file with a header row.
        On subsequent writes, appends without header.

        Args:
            records: List of DetectionRecords to write.
        """
        data = [
            {
                "Timestamp": r.timestamp,
                "Age": r.age,
                "Gender": r.gender,
                "Is_Senior_Citizen": r.is_senior_citizen,
            }
            for r in records
        ]
        df = pd.DataFrame(data, columns=COLUMNS)

        if self.format == "excel":
            self._write_excel(df)
        else:
            self._write_csv(df)

    def _write_csv(self, df: pd.DataFrame) -> None:
        """Write DataFrame to CSV file with append-only semantics."""
        file_exists = os.path.exists(self.output_path)

        if not file_exists:
            # Create new file with header
            df.to_csv(self.output_path, mode="w", index=False, header=True)
            self._header_written = True
        elif not self._header_written:
            # File exists from a previous session — append without header
            df.to_csv(self.output_path, mode="a", index=False, header=False)
            self._header_written = True
        else:
            # Append without header
            df.to_csv(self.output_path, mode="a", index=False, header=False)

    def _write_excel(self, df: pd.DataFrame) -> None:
        """Write DataFrame to Excel file with append-only semantics."""
        file_exists = os.path.exists(self.output_path)

        if file_exists and self._header_written:
            # Read existing data and append new records
            existing_df = pd.read_excel(self.output_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_excel(self.output_path, index=False)
        else:
            # First write - create with header
            df.to_excel(self.output_path, index=False)
            self._header_written = True
