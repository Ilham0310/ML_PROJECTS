"""Prediction result dataclass for Nationality Detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PredictionResult:
    """Final output from the nationality inference pipeline."""

    nationality: str
    nationality_confidence: float
    emotion: str
    emotion_confidence: float
    age: Optional[int] = None
    dress_colour: Optional[str] = None
    dress_colour_confidence: Optional[float] = None
