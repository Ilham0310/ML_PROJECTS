"""Image display utilities."""

from __future__ import annotations

from typing import Tuple


def scale_to_fit(
    image_width: int,
    image_height: int,
    max_width: int,
    max_height: int,
) -> Tuple[int, int]:
    """Return dimensions that fit inside bounds while preserving aspect ratio."""

    if min(image_width, image_height, max_width, max_height) <= 0:
        raise ValueError("all dimensions must be positive")
    scale = min(max_width / image_width, max_height / image_height)
    scaled_width = max(1, int(round(image_width * scale)))
    scaled_height = max(1, int(round(image_height * scale)))
    return min(scaled_width, max_width), min(scaled_height, max_height)
