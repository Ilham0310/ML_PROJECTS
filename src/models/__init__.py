"""
Models package for the Long-Hair Gender Identification system.

Exports the three model components and shared exceptions.
"""

from .age_estimator import AgeEstimator, ModelLoadError
from .dress_colour_classifier import DressColourClassifier
from .emotion_predictor import EmotionPredictor
from .nationality_age_estimator import NationalityAgeEstimator
from .nationality_detector import NationalityDetector
from .voice_age_estimator import VoiceAgeEstimator
from .voice_emotion_detector import VoiceEmotionDetector
from .voice_gender_classifier import VoiceGenderClassifier
from .voice_model_utils import VoiceModelLoadError
from src.car_colour.models import (
    AnnotationConfig,
    BoundingBox,
    CarDetectionResult,
    ColourClassifierConfig,
    Detection,
    DetectionResult,
)

__all__ = [
    'AnnotationConfig',
    'AgeEstimator',
    'BoundingBox',
    'CarDetectionResult',
    'ColourClassifierConfig',
    'Detection',
    'DetectionResult',
    'DressColourClassifier',
    'EmotionPredictor',
    'ModelLoadError',
    'NationalityAgeEstimator',
    'NationalityDetector',
    'VoiceAgeEstimator',
    'VoiceEmotionDetector',
    'VoiceGenderClassifier',
    'VoiceModelLoadError',
]
