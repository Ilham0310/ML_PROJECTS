"""Property tests for Age-Emotion Voice Detection."""

from unittest.mock import MagicMock

import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

from src.audio.feature_extractor import FeatureExtractor
from src.audio.voice_processor import VoiceProcessor
from src.inference.voice_decision_router import VoiceDecisionRouter
from src.models.voice_age_estimator import VoiceAgeEstimator
from src.models.voice_emotion_detector import VoiceEmotionDetector
from src.models.voice_gender_classifier import VoiceGenderClassifier


feature_vectors = arrays(
    np.float32,
    (28,),
    elements=st.floats(-3, 3, allow_nan=False, allow_infinity=False),
)


# Feature: age-emotion-voice-detection, Property 1: Format validation accepts valid and rejects invalid extensions
@given(ext=st.sampled_from([".wav", ".mp3", ".flac", ".ogg", ".txt", ".pdf", ".exe", ""]))
@settings(max_examples=100)
def test_voice_format_validation_property(ext):
    is_valid, _ = VoiceProcessor().validate_format(f"voice{ext}")
    assert is_valid is (ext.lower() in VoiceProcessor.SUPPORTED_FORMATS)


# Feature: age-emotion-voice-detection, Property 2: Resampling produces consistent sample rate
@given(
    original_sr=st.integers(8000, 48000),
    audio=arrays(np.float32, (256,), elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False)),
)
@settings(max_examples=100, deadline=None)
def test_voice_resampling_sample_rate_property(original_sr, audio):
    output = VoiceProcessor.resample(audio, original_sr, VoiceProcessor.TARGET_SAMPLE_RATE)
    expected_len = max(1, int(round(audio.size / original_sr * VoiceProcessor.TARGET_SAMPLE_RATE)))
    assert output.dtype == np.float32
    assert output.shape == (expected_len,)


# Feature: age-emotion-voice-detection, Property 3: Feature vector has correct structure and dimensionality
@given(audio=arrays(np.float32, (128,), elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False)))
@settings(max_examples=100, deadline=None)
def test_voice_feature_vector_dimensionality_property(audio):
    features = FeatureExtractor(prefer_librosa=False).extract(audio, sample_rate=128)
    assert features.shape == (28,)


# Feature: age-emotion-voice-detection, Property 4: Feature normalization produces zero mean and unit variance
@given(values=arrays(np.float32, (28,), elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)))
@settings(max_examples=100, deadline=None)
def test_voice_feature_normalization_property(values):
    normalized = FeatureExtractor.normalize(values)
    assert abs(float(np.mean(normalized))) <= 1e-5
    if float(np.std(values.astype(np.float64))) > 1e-5:
        assert abs(float(np.std(normalized)) - 1.0) <= 1e-5


# Feature: age-emotion-voice-detection, Property 5: Gender classifier output domain
@given(features=feature_vectors)
@settings(max_examples=100, deadline=None)
def test_voice_gender_output_domain_property(features):
    assert VoiceGenderClassifier().predict(features) in {"male", "female"}


# Feature: age-emotion-voice-detection, Property 6: Gender-based routing correctness
@given(gender=st.sampled_from(["male", "female"]))
@settings(max_examples=100)
def test_voice_gender_routing_property(gender):
    router = _router_with_mocks(gender=gender, age=50)
    result = router.process("voice.wav")
    if gender == "female":
        router.age_estimator.predict.assert_not_called()
        router.emotion_detector.predict.assert_not_called()
        assert result.message == "Upload male voice"
    else:
        router.age_estimator.predict.assert_called_once()


# Feature: age-emotion-voice-detection, Property 7: Age estimate range invariant
@given(features=feature_vectors)
@settings(max_examples=100, deadline=None)
def test_voice_age_range_property(features):
    age = VoiceAgeEstimator().predict(features)
    assert isinstance(age, int)
    assert 10 <= age <= 100


# Feature: age-emotion-voice-detection, Property 8: Age-threshold routing correctness
@given(age=st.integers(10, 100))
@settings(max_examples=100)
def test_voice_age_threshold_routing_property(age):
    router = _router_with_mocks(gender="male", age=age)
    result = router.process("voice.wav")
    if age > 60:
        router.emotion_detector.predict.assert_called_once()
        assert result.is_senior_citizen is True
    else:
        router.emotion_detector.predict.assert_not_called()
        assert result.is_senior_citizen is False


# Feature: age-emotion-voice-detection, Property 9: Emotion detector output domain
@given(features=feature_vectors)
@settings(max_examples=100, deadline=None)
def test_voice_emotion_output_domain_property(features):
    assert VoiceEmotionDetector().predict(features) in set(VoiceEmotionDetector.EMOTIONS)


# Feature: age-emotion-voice-detection, Property 10: Pipeline execution order
def test_voice_pipeline_execution_order_property():
    calls = []
    router = _router_with_mocks(gender="male", age=70, calls=calls)
    router.process("voice.wav")
    assert calls == ["load", "extract", "gender", "age", "emotion"]


# Feature: age-emotion-voice-detection, Property 11: Error halts pipeline
@given(step=st.sampled_from(["load", "extract", "gender", "age", "emotion"]))
@settings(max_examples=20)
def test_voice_error_halts_pipeline_property(step):
    router = _router_with_mocks(gender="male", age=70, fail_step=step)
    result = router.process("voice.wav")
    assert result.error is not None
    if step in {"load", "extract", "gender"}:
        router.age_estimator.predict.assert_not_called()
    if step in {"load", "extract", "gender", "age"}:
        router.emotion_detector.predict.assert_not_called()


# Feature: age-emotion-voice-detection, Property 12: Feature extraction happens exactly once
def test_voice_feature_extraction_once_property():
    feature_vector = np.arange(28, dtype=np.float32)
    router = _router_with_mocks(gender="male", age=70, features=feature_vector)
    router.process("voice.wav")
    router.feature_extractor.extract.assert_called_once()
    router.gender_classifier.predict.assert_called_once_with(feature_vector)
    router.age_estimator.predict.assert_called_once_with(feature_vector)
    router.emotion_detector.predict.assert_called_once_with(feature_vector)


def _router_with_mocks(
    gender="male",
    age=70,
    features=None,
    calls=None,
    fail_step=None,
):
    calls = calls if calls is not None else []
    features = features if features is not None else np.ones(28, dtype=np.float32)

    voice_processor = MagicMock()
    feature_extractor = MagicMock()
    gender_classifier = MagicMock()
    age_estimator = MagicMock()
    emotion_detector = MagicMock()

    def load(_):
        calls.append("load")
        if fail_step == "load":
            raise ValueError("load failed")
        return np.ones(128, dtype=np.float32), 128

    def extract(_, __):
        calls.append("extract")
        if fail_step == "extract":
            raise ValueError("extract failed")
        return features

    def predict_gender(_):
        calls.append("gender")
        if fail_step == "gender":
            raise RuntimeError("gender failed")
        return gender

    def predict_age(_):
        calls.append("age")
        if fail_step == "age":
            raise RuntimeError("age failed")
        return age

    def predict_emotion(_):
        calls.append("emotion")
        if fail_step == "emotion":
            raise RuntimeError("emotion failed")
        return "happy"

    voice_processor.load.side_effect = load
    feature_extractor.extract.side_effect = extract
    gender_classifier.predict.side_effect = predict_gender
    age_estimator.predict.side_effect = predict_age
    emotion_detector.predict.side_effect = predict_emotion

    return VoiceDecisionRouter(
        voice_processor=voice_processor,
        feature_extractor=feature_extractor,
        gender_classifier=gender_classifier,
        age_estimator=age_estimator,
        emotion_detector=emotion_detector,
    )
