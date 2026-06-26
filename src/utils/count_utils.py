"""Counting utilities for detection results."""

from __future__ import annotations

from typing import Iterable


def count_by_class(detections: Iterable[object], target_class: int) -> int:
    """Count detection objects whose class id equals ``target_class``."""

    count = 0
    for detection in detections:
        class_id = getattr(detection, "class_id", None)
        if class_id is None and hasattr(detection, "bbox"):
            class_id = getattr(detection.bbox, "class_id", None)
        if class_id == target_class:
            count += 1
    return count
