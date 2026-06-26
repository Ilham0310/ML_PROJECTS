"""Senior citizen classification routing module.

Applies age and confidence thresholds to classify detected individuals
as Senior Citizen or Non-Senior, with appropriate color coding and labels.
"""

from typing import Tuple

from src.senior.models import ClassificationResult


class SeniorRouter:
    """Routes age/gender predictions into classification results.

    Applies threshold logic to determine if a detected person is a senior
    citizen, assigns appropriate bounding box colors, and generates display
    labels.

    Constants:
        AGE_THRESHOLD: Age above which a person is classified as senior (60).
        AGE_CONFIDENCE_THRESHOLD: Minimum age confidence for reliable classification (0.3).
        GENDER_CONFIDENCE_THRESHOLD: Minimum gender confidence to display gender (0.4).
        COLOR_SENIOR: Green BGR color for senior citizens.
        COLOR_NON_SENIOR: Blue BGR color for non-senior individuals.
        COLOR_LOW_CONF: Yellow BGR color for low-confidence estimates.
    """

    AGE_THRESHOLD: int = 60
    AGE_CONFIDENCE_THRESHOLD: float = 0.3
    GENDER_CONFIDENCE_THRESHOLD: float = 0.4

    COLOR_SENIOR: Tuple[int, int, int] = (0, 255, 0)       # Green (BGR)
    COLOR_NON_SENIOR: Tuple[int, int, int] = (255, 0, 0)   # Blue (BGR)
    COLOR_LOW_CONF: Tuple[int, int, int] = (0, 255, 255)   # Yellow (BGR)

    def route(
        self, age: int, age_conf: float, gender: str, gender_conf: float
    ) -> ClassificationResult:
        """Apply threshold logic to produce a classification result.

        Args:
            age: Estimated age as integer in [1, 100].
            age_conf: Age estimation confidence in [0.0, 1.0].
            gender: Raw gender prediction ("Male" or "Female").
            gender_conf: Gender classification confidence in [0.0, 1.0].

        Returns:
            ClassificationResult with classification flags, display values,
            box color, and label text.
        """
        # Determine display gender based on gender confidence
        if gender_conf < self.GENDER_CONFIDENCE_THRESHOLD:
            display_gender = "Unknown"
        else:
            display_gender = gender

        # Route based on age confidence and age value
        if age_conf < self.AGE_CONFIDENCE_THRESHOLD:
            # Low confidence path
            is_senior = False
            is_low_confidence = True
            box_color = self.COLOR_LOW_CONF
            label_text = f"Age: {age} (low confidence) | {display_gender}"
        elif age > self.AGE_THRESHOLD:
            # Senior citizen path
            is_senior = True
            is_low_confidence = False
            box_color = self.COLOR_SENIOR
            label_text = f"Senior Citizen | Age: {age} | {display_gender}"
        else:
            # Non-senior path
            is_senior = False
            is_low_confidence = False
            box_color = self.COLOR_NON_SENIOR
            label_text = f"Age: {age} | {display_gender}"

        return ClassificationResult(
            is_senior=is_senior,
            is_low_confidence=is_low_confidence,
            display_age=age,
            display_gender=display_gender,
            box_color=box_color,
            label_text=label_text,
        )
