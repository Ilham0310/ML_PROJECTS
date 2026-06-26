"""Decision routing for Age-Emotion Voice Detection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from src.audio.feature_extractor import FeatureExtractor
from src.audio.voice_processor import VoiceProcessor
from src.models.voice_age_estimator import VoiceAgeEstimator
from src.models.voice_emotion_detector import VoiceEmotionDetector
from src.models.voice_gender_classifier import VoiceGenderClassifier
from src.models.voice_model_utils import VoiceModelLoadError


@dataclass
class VoicePredictionResult:
    """Result returned by the voice decision pipeline."""

    gender: Optional[str] = None
    age: Optional[int] = None
    is_senior_citizen: bool = False
    emotion: Optional[str] = None
    message: str = ""
    error: Optional[str] = None


class VoiceDecisionRouter:
    """Orchestrates voice loading, features, gender, age, and emotion."""

    SENIOR_AGE_THRESHOLD = 60

    def __init__(
        self,
        models_dir: str = "models",
        voice_processor: Optional[VoiceProcessor] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        gender_classifier: Optional[VoiceGenderClassifier] = None,
        age_estimator: Optional[VoiceAgeEstimator] = None,
        emotion_detector: Optional[VoiceEmotionDetector] = None,
    ) -> None:
        self.models_dir = models_dir
        self.voice_processor = voice_processor or VoiceProcessor()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.gender_classifier = gender_classifier or VoiceGenderClassifier()
        self.age_estimator = age_estimator or VoiceAgeEstimator()
        self.emotion_detector = emotion_detector or VoiceEmotionDetector()

    def load_models(self) -> None:
        """Load all voice model files from ``models_dir``."""

        self.gender_classifier.load(
            os.path.join(self.models_dir, "voice_gender_classifier.keras")
        )
        self.age_estimator.load(os.path.join(self.models_dir, "voice_age_estimator.keras"))
        self.emotion_detector.load(
            os.path.join(self.models_dir, "voice_emotion_detector.keras")
        )

    def process(self, file_path: str) -> VoicePredictionResult:
        """Run the full voice analysis pipeline and halt on first error."""

        try:
            audio, sample_rate = self.voice_processor.load(file_path)
        except Exception as exc:
            return VoicePredictionResult(message=str(exc), error=str(exc))

        try:
            features = self.feature_extractor.extract(audio, sample_rate)
        except Exception as exc:
            return VoicePredictionResult(message=str(exc), error=str(exc))

        try:
            gender = self.gender_classifier.predict(features)
        except Exception:
            message = "Gender classification failed."
            return VoicePredictionResult(message=message, error=message)

        if gender == "female":
            return VoicePredictionResult(gender="female", message="Upload male voice")

        try:
            age = self.age_estimator.predict(features)
        except Exception:
            message = "Age estimation failed."
            return VoicePredictionResult(gender="male", message=message, error=message)

        if age <= self.SENIOR_AGE_THRESHOLD:
            return VoicePredictionResult(
                gender="male",
                age=age,
                is_senior_citizen=False,
                message=f"Estimated age: {age}",
            )

        try:
            emotion = self.emotion_detector.predict(features)
        except Exception:
            message = "Emotion detection failed."
            return VoicePredictionResult(
                gender="male",
                age=age,
                is_senior_citizen=True,
                message=message,
                error=message,
            )

        return VoicePredictionResult(
            gender="male",
            age=age,
            is_senior_citizen=True,
            emotion=emotion,
            message=f"Estimated age: {age}. Senior Citizen. Emotion: {emotion}",
        )
