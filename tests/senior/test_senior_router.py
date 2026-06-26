"""Unit tests for the SeniorRouter class.

Validates all routing conditions:
- Low age confidence → Non-Senior, yellow box, "(low confidence)" in label
- High confidence + age > 60 → Senior Citizen, green box
- High confidence + age <= 60 → Non-Senior, blue box
- Low gender confidence → display gender as "Unknown"
- Label text formatting for all cases

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 4.5
"""

import pytest

from src.senior.senior_router import SeniorRouter
from src.senior.models import ClassificationResult


@pytest.fixture
def router() -> SeniorRouter:
    """Create a SeniorRouter instance."""
    return SeniorRouter()


class TestSeniorRouterConstants:
    """Tests that SeniorRouter constants are set correctly."""

    def test_age_threshold_is_60(self, router: SeniorRouter):
        assert router.AGE_THRESHOLD == 60

    def test_age_confidence_threshold_is_0_3(self, router: SeniorRouter):
        assert router.AGE_CONFIDENCE_THRESHOLD == 0.3

    def test_gender_confidence_threshold_is_0_4(self, router: SeniorRouter):
        assert router.GENDER_CONFIDENCE_THRESHOLD == 0.4

    def test_color_senior_is_green(self, router: SeniorRouter):
        assert router.COLOR_SENIOR == (0, 255, 0)

    def test_color_non_senior_is_blue(self, router: SeniorRouter):
        assert router.COLOR_NON_SENIOR == (255, 0, 0)

    def test_color_low_conf_is_yellow(self, router: SeniorRouter):
        assert router.COLOR_LOW_CONF == (0, 255, 255)


class TestLowAgeConfidence:
    """Tests routing when age confidence is below threshold (< 0.3).

    Validates: Requirement 5.6 — low confidence → Non_Senior, yellow box,
    "(low confidence)" in label.
    """

    def test_low_confidence_returns_non_senior(self, router: SeniorRouter):
        result = router.route(age=72, age_conf=0.2, gender="Female", gender_conf=0.8)
        assert result.is_senior is False

    def test_low_confidence_flags_low_confidence(self, router: SeniorRouter):
        result = router.route(age=72, age_conf=0.1, gender="Male", gender_conf=0.9)
        assert result.is_low_confidence is True

    def test_low_confidence_uses_yellow_box(self, router: SeniorRouter):
        result = router.route(age=65, age_conf=0.29, gender="Female", gender_conf=0.7)
        assert result.box_color == (0, 255, 255)

    def test_low_confidence_label_contains_low_confidence_text(self, router: SeniorRouter):
        result = router.route(age=80, age_conf=0.0, gender="Male", gender_conf=0.9)
        assert "(low confidence)" in result.label_text

    def test_low_confidence_label_does_not_contain_senior_citizen(self, router: SeniorRouter):
        result = router.route(age=90, age_conf=0.25, gender="Female", gender_conf=0.8)
        assert "Senior Citizen" not in result.label_text

    def test_low_confidence_at_zero(self, router: SeniorRouter):
        result = router.route(age=50, age_conf=0.0, gender="Male", gender_conf=0.5)
        assert result.is_senior is False
        assert result.is_low_confidence is True
        assert result.box_color == (0, 255, 255)

    def test_low_confidence_preserves_display_age(self, router: SeniorRouter):
        result = router.route(age=45, age_conf=0.1, gender="Female", gender_conf=0.8)
        assert result.display_age == 45

    def test_low_confidence_with_young_age(self, router: SeniorRouter):
        result = router.route(age=25, age_conf=0.2, gender="Male", gender_conf=0.6)
        assert result.is_senior is False
        assert result.is_low_confidence is True
        assert result.box_color == (0, 255, 255)


class TestSeniorCitizenClassification:
    """Tests routing when age > 60 and age confidence >= 0.3.

    Validates: Requirements 5.1, 5.3 — Senior_Citizen, green box,
    "Senior Citizen" in label.
    """

    def test_senior_returns_is_senior_true(self, router: SeniorRouter):
        result = router.route(age=61, age_conf=0.5, gender="Female", gender_conf=0.8)
        assert result.is_senior is True

    def test_senior_is_not_low_confidence(self, router: SeniorRouter):
        result = router.route(age=75, age_conf=0.9, gender="Male", gender_conf=0.7)
        assert result.is_low_confidence is False

    def test_senior_uses_green_box(self, router: SeniorRouter):
        result = router.route(age=70, age_conf=0.6, gender="Female", gender_conf=0.9)
        assert result.box_color == (0, 255, 0)

    def test_senior_label_contains_senior_citizen(self, router: SeniorRouter):
        result = router.route(age=72, age_conf=0.8, gender="Female", gender_conf=0.7)
        assert "Senior Citizen" in result.label_text

    def test_senior_label_contains_age(self, router: SeniorRouter):
        result = router.route(age=72, age_conf=0.8, gender="Female", gender_conf=0.7)
        assert "72" in result.label_text

    def test_senior_label_contains_gender(self, router: SeniorRouter):
        result = router.route(age=72, age_conf=0.8, gender="Female", gender_conf=0.7)
        assert "Female" in result.label_text

    def test_senior_at_boundary_age_61(self, router: SeniorRouter):
        result = router.route(age=61, age_conf=0.3, gender="Male", gender_conf=0.5)
        assert result.is_senior is True
        assert result.box_color == (0, 255, 0)

    def test_senior_at_minimum_confidence_threshold(self, router: SeniorRouter):
        result = router.route(age=65, age_conf=0.3, gender="Female", gender_conf=0.8)
        assert result.is_senior is True
        assert result.is_low_confidence is False

    def test_senior_preserves_display_age(self, router: SeniorRouter):
        result = router.route(age=88, age_conf=0.95, gender="Male", gender_conf=0.9)
        assert result.display_age == 88

    def test_senior_at_age_100(self, router: SeniorRouter):
        result = router.route(age=100, age_conf=0.7, gender="Female", gender_conf=0.6)
        assert result.is_senior is True
        assert result.box_color == (0, 255, 0)


class TestNonSeniorClassification:
    """Tests routing when age <= 60 and age confidence >= 0.3.

    Validates: Requirements 5.2, 5.4 — Non_Senior, blue box,
    no "Senior Citizen" text.
    """

    def test_non_senior_returns_is_senior_false(self, router: SeniorRouter):
        result = router.route(age=35, age_conf=0.8, gender="Male", gender_conf=0.9)
        assert result.is_senior is False

    def test_non_senior_is_not_low_confidence(self, router: SeniorRouter):
        result = router.route(age=45, age_conf=0.6, gender="Female", gender_conf=0.7)
        assert result.is_low_confidence is False

    def test_non_senior_uses_blue_box(self, router: SeniorRouter):
        result = router.route(age=30, age_conf=0.7, gender="Male", gender_conf=0.8)
        assert result.box_color == (255, 0, 0)

    def test_non_senior_label_does_not_contain_senior_citizen(self, router: SeniorRouter):
        result = router.route(age=55, age_conf=0.9, gender="Female", gender_conf=0.7)
        assert "Senior Citizen" not in result.label_text

    def test_non_senior_label_contains_age(self, router: SeniorRouter):
        result = router.route(age=40, age_conf=0.7, gender="Male", gender_conf=0.8)
        assert "40" in result.label_text

    def test_non_senior_label_contains_gender(self, router: SeniorRouter):
        result = router.route(age=40, age_conf=0.7, gender="Male", gender_conf=0.8)
        assert "Male" in result.label_text

    def test_non_senior_at_boundary_age_60(self, router: SeniorRouter):
        """Age exactly 60 should be Non_Senior (threshold is >60, not >=60)."""
        result = router.route(age=60, age_conf=0.5, gender="Female", gender_conf=0.8)
        assert result.is_senior is False
        assert result.box_color == (255, 0, 0)

    def test_non_senior_preserves_display_age(self, router: SeniorRouter):
        result = router.route(age=22, age_conf=0.85, gender="Male", gender_conf=0.9)
        assert result.display_age == 22

    def test_non_senior_at_age_1(self, router: SeniorRouter):
        result = router.route(age=1, age_conf=0.4, gender="Female", gender_conf=0.6)
        assert result.is_senior is False
        assert result.box_color == (255, 0, 0)


class TestGenderConfidenceRouting:
    """Tests gender confidence threshold logic.

    Validates: Requirement 4.5 — gender_conf < 0.4 → display as "Unknown".
    """

    def test_low_gender_confidence_displays_unknown(self, router: SeniorRouter):
        result = router.route(age=50, age_conf=0.8, gender="Female", gender_conf=0.3)
        assert result.display_gender == "Unknown"

    def test_low_gender_confidence_label_contains_unknown(self, router: SeniorRouter):
        result = router.route(age=50, age_conf=0.8, gender="Male", gender_conf=0.1)
        assert "Unknown" in result.label_text

    def test_high_gender_confidence_displays_actual_gender(self, router: SeniorRouter):
        result = router.route(age=50, age_conf=0.8, gender="Female", gender_conf=0.5)
        assert result.display_gender == "Female"

    def test_gender_confidence_at_threshold_displays_actual(self, router: SeniorRouter):
        """Gender confidence exactly at 0.4 should display the actual gender."""
        result = router.route(age=50, age_conf=0.8, gender="Male", gender_conf=0.4)
        assert result.display_gender == "Male"

    def test_gender_confidence_just_below_threshold(self, router: SeniorRouter):
        result = router.route(age=50, age_conf=0.8, gender="Male", gender_conf=0.39)
        assert result.display_gender == "Unknown"

    def test_low_gender_confidence_with_senior(self, router: SeniorRouter):
        result = router.route(age=75, age_conf=0.9, gender="Female", gender_conf=0.2)
        assert result.is_senior is True
        assert result.display_gender == "Unknown"
        assert "Unknown" in result.label_text

    def test_low_gender_confidence_with_low_age_confidence(self, router: SeniorRouter):
        result = router.route(age=65, age_conf=0.1, gender="Male", gender_conf=0.1)
        assert result.display_gender == "Unknown"
        assert result.is_low_confidence is True

    def test_zero_gender_confidence_displays_unknown(self, router: SeniorRouter):
        result = router.route(age=40, age_conf=0.5, gender="Female", gender_conf=0.0)
        assert result.display_gender == "Unknown"


class TestClassificationResultFields:
    """Tests that ClassificationResult dataclass fields are populated correctly."""

    def test_returns_classification_result_type(self, router: SeniorRouter):
        result = router.route(age=50, age_conf=0.8, gender="Male", gender_conf=0.9)
        assert isinstance(result, ClassificationResult)

    def test_senior_full_result(self, router: SeniorRouter):
        result = router.route(age=72, age_conf=0.85, gender="Female", gender_conf=0.9)
        assert result.is_senior is True
        assert result.is_low_confidence is False
        assert result.display_age == 72
        assert result.display_gender == "Female"
        assert result.box_color == (0, 255, 0)
        assert "Senior Citizen" in result.label_text
        assert "72" in result.label_text
        assert "Female" in result.label_text

    def test_non_senior_full_result(self, router: SeniorRouter):
        result = router.route(age=35, age_conf=0.7, gender="Male", gender_conf=0.8)
        assert result.is_senior is False
        assert result.is_low_confidence is False
        assert result.display_age == 35
        assert result.display_gender == "Male"
        assert result.box_color == (255, 0, 0)
        assert "Senior Citizen" not in result.label_text
        assert "35" in result.label_text
        assert "Male" in result.label_text

    def test_low_confidence_full_result(self, router: SeniorRouter):
        result = router.route(age=68, age_conf=0.15, gender="Female", gender_conf=0.8)
        assert result.is_senior is False
        assert result.is_low_confidence is True
        assert result.display_age == 68
        assert result.display_gender == "Female"
        assert result.box_color == (0, 255, 255)
        assert "(low confidence)" in result.label_text
        assert "68" in result.label_text


class TestLabelTextFormatting:
    """Tests that label text is properly formatted for display."""

    def test_senior_label_format(self, router: SeniorRouter):
        result = router.route(age=72, age_conf=0.8, gender="Female", gender_conf=0.7)
        assert result.label_text == "Senior Citizen | Age: 72 | Female"

    def test_non_senior_label_format(self, router: SeniorRouter):
        result = router.route(age=35, age_conf=0.7, gender="Male", gender_conf=0.8)
        assert result.label_text == "Age: 35 | Male"

    def test_low_confidence_label_format(self, router: SeniorRouter):
        result = router.route(age=55, age_conf=0.2, gender="Female", gender_conf=0.8)
        assert result.label_text == "Age: 55 (low confidence) | Female"

    def test_senior_with_unknown_gender_label_format(self, router: SeniorRouter):
        result = router.route(age=70, age_conf=0.6, gender="Male", gender_conf=0.2)
        assert result.label_text == "Senior Citizen | Age: 70 | Unknown"

    def test_non_senior_with_unknown_gender_label_format(self, router: SeniorRouter):
        result = router.route(age=40, age_conf=0.5, gender="Female", gender_conf=0.1)
        assert result.label_text == "Age: 40 | Unknown"

    def test_low_confidence_with_unknown_gender_label_format(self, router: SeniorRouter):
        result = router.route(age=80, age_conf=0.1, gender="Male", gender_conf=0.2)
        assert result.label_text == "Age: 80 (low confidence) | Unknown"
