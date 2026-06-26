"""Conditional decision routing for Nationality Detection."""

from __future__ import annotations

import numpy as np

from .age_estimator import NationalityAgeEstimator
from .dress_colour_classifier import DressColourClassifier
from .emotion_predictor import EmotionPredictor
from .prediction_result import PredictionResult
from .preprocessor import NationalityPreprocessor


class NationalityDecisionRouter:
    """Invokes conditional predictors based on detected nationality."""

    AGE_NATIONALITIES = {"Indian", "US/American"}
    DRESS_NATIONALITIES = {"Indian", "African"}

    def __init__(self, preprocessor: NationalityPreprocessor | None = None) -> None:
        self.preprocessor = preprocessor or NationalityPreprocessor()

    def _emotion_input(self, face_image: np.ndarray) -> np.ndarray:
        if face_image.shape == EmotionPredictor.INPUT_SHAPE:
            return face_image.astype(np.float32)
        return self.preprocessor.preprocess_face_grayscale(face_image, (48, 48))

    def _age_input(self, face_image: np.ndarray) -> np.ndarray:
        if face_image.shape == NationalityAgeEstimator.INPUT_SHAPE:
            return face_image.astype(np.float32)
        return self.preprocessor.preprocess_face(face_image, (128, 128))

    def _dress_input(self, full_image: np.ndarray) -> np.ndarray:
        if full_image.shape == DressColourClassifier.INPUT_SHAPE:
            return full_image.astype(np.float32)
        return self.preprocessor.preprocess_full_image(full_image, (128, 128))

    def route(
        self,
        face_image: np.ndarray,
        full_image: np.ndarray,
        nationality: str,
        nationality_confidence: float,
        emotion_predictor: EmotionPredictor,
        age_estimator: NationalityAgeEstimator,
        dress_colour_classifier: DressColourClassifier,
    ) -> PredictionResult:
        """Return a result with fields populated according to the routing table."""

        emotion, emotion_confidence = emotion_predictor.predict(
            self._emotion_input(face_image)
        )

        age = None
        dress_colour = None
        dress_colour_confidence = None

        if nationality in self.AGE_NATIONALITIES:
            age = age_estimator.predict(self._age_input(face_image))

        if nationality in self.DRESS_NATIONALITIES:
            dress_colour, dress_colour_confidence = dress_colour_classifier.predict(
                self._dress_input(full_image)
            )

        return PredictionResult(
            nationality=nationality,
            nationality_confidence=float(nationality_confidence),
            emotion=emotion,
            emotion_confidence=float(emotion_confidence),
            age=age,
            dress_colour=dress_colour,
            dress_colour_confidence=dress_colour_confidence,
        )
