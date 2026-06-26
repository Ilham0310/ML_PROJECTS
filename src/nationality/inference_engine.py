"""Inference engine for Nationality Detection."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from .age_estimator import NationalityAgeEstimator
from .decision_router import NationalityDecisionRouter
from .dress_colour_classifier import DressColourClassifier
from .emotion_predictor import EmotionPredictor
from .exceptions import (
    CORRUPT_IMAGE_MESSAGE,
    FILE_SIZE_MESSAGE,
    GENERIC_INFERENCE_MESSAGE,
    INVALID_FORMAT_MESSAGE,
    MODEL_LOAD_MESSAGE,
    MULTIPLE_FACES_MESSAGE,
    NO_FACE_MESSAGE,
    CorruptImageError,
    FileSizeError,
    InferenceError,
    InvalidFileFormatError,
    ModelLoadError,
    MultipleFacesError,
    NoFaceDetectedError,
)
from .nationality_detector import NationalityDetector
from .prediction_result import PredictionResult
from .preprocessor import NationalityPreprocessor

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class NationalityInferenceEngine:
    """Coordinates validation, face detection, prediction, and routing."""

    MODEL_FILES = {
        "nationality_detector": "nationality_detector.keras",
        "emotion_predictor": "emotion_predictor.keras",
        "age_estimator": "age_estimator.keras",
        "dress_colour_classifier": "dress_colour_classifier.keras",
    }

    def __init__(
        self,
        model_dir: str = "models",
        nationality_detector: Optional[NationalityDetector] = None,
        emotion_predictor: Optional[EmotionPredictor] = None,
        age_estimator: Optional[NationalityAgeEstimator] = None,
        dress_colour_classifier: Optional[DressColourClassifier] = None,
        router: Optional[NationalityDecisionRouter] = None,
    ) -> None:
        self.model_dir = model_dir
        self.nationality_detector = nationality_detector
        self.emotion_predictor = emotion_predictor
        self.age_estimator = age_estimator
        self.dress_colour_classifier = dress_colour_classifier
        self.router = router or NationalityDecisionRouter()
        self.preprocessor = NationalityPreprocessor()
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def load_models(self, model_dir: Optional[str] = None) -> None:
        """Load all four trained model files from ``model_dir``."""

        if model_dir is not None:
            self.model_dir = model_dir
        missing = [
            filename
            for filename in self.MODEL_FILES.values()
            if not os.path.isfile(os.path.join(self.model_dir, filename))
        ]
        if missing:
            raise ModelLoadError(f"{MODEL_LOAD_MESSAGE} Missing: {', '.join(missing)}")

        try:
            self.nationality_detector = NationalityDetector()
            self.nationality_detector.load(
                os.path.join(self.model_dir, self.MODEL_FILES["nationality_detector"])
            )
            self.emotion_predictor = EmotionPredictor()
            self.emotion_predictor.load(
                os.path.join(self.model_dir, self.MODEL_FILES["emotion_predictor"])
            )
            self.age_estimator = NationalityAgeEstimator()
            self.age_estimator.load(
                os.path.join(self.model_dir, self.MODEL_FILES["age_estimator"])
            )
            self.dress_colour_classifier = DressColourClassifier()
            self.dress_colour_classifier.load(
                os.path.join(self.model_dir, self.MODEL_FILES["dress_colour_classifier"])
            )
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(MODEL_LOAD_MESSAGE) from exc

    def predict(self, image_path: str) -> PredictionResult:
        """Run the complete nationality prediction pipeline."""

        self._validate_file_format(image_path)
        self._validate_file_size(image_path)
        image = self._load_image(image_path)
        face_roi = self._detect_faces(image)

        if any(
            model is None
            for model in [
                self.nationality_detector,
                self.emotion_predictor,
                self.age_estimator,
                self.dress_colour_classifier,
            ]
        ):
            self.load_models()

        try:
            nationality_input = self.preprocessor.preprocess_face(face_roi, (128, 128))
            nationality, nationality_confidence = self.nationality_detector.predict(
                nationality_input
            )
            return self.router.route(
                face_image=face_roi,
                full_image=image,
                nationality=nationality,
                nationality_confidence=nationality_confidence,
                emotion_predictor=self.emotion_predictor,
                age_estimator=self.age_estimator,
                dress_colour_classifier=self.dress_colour_classifier,
            )
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(GENERIC_INFERENCE_MESSAGE) from exc

    def _validate_file_format(self, path: str) -> None:
        if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
            raise InvalidFileFormatError(INVALID_FORMAT_MESSAGE)

    def _validate_file_size(self, path: str) -> None:
        if not os.path.isfile(path):
            raise CorruptImageError(CORRUPT_IMAGE_MESSAGE)
        if os.path.getsize(path) > MAX_FILE_SIZE_BYTES:
            raise FileSizeError(FILE_SIZE_MESSAGE)

    def _load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        if image is None:
            raise CorruptImageError(CORRUPT_IMAGE_MESSAGE)
        return image

    def _detect_faces(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        count = len(faces)
        if count == 0:
            raise NoFaceDetectedError(NO_FACE_MESSAGE)
        if count > 1:
            raise MultipleFacesError(MULTIPLE_FACES_MESSAGE)
        x, y, w, h = [int(v) for v in faces[0]]
        return image_bgr[y : y + h, x : x + w].copy()
