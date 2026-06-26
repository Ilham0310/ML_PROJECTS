"""Simple end-user dashboard for all ML project specs.

Run with:
    streamlit run submission_dashboard.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "models" / "submission"
UPLOAD_ROOT = PROJECT_ROOT / "data" / "submission_dashboard_uploads"
SAMPLE_ROOT = PROJECT_ROOT / "data" / "dashboard_samples"
SAMPLE_MANIFEST = SAMPLE_ROOT / "samples_manifest.json"


@dataclass(frozen=True)
class SpecConfig:
    key: str
    title: str
    zip_name: str
    expected_files: tuple[str, ...]
    file_types: tuple[str, ...]
    input_label: str
    description: str


SPECS: dict[str, SpecConfig] = {
    "long_hair": SpecConfig(
        key="long_hair",
        title="Long-Hair Gender",
        zip_name="long_hair_gender_models.zip",
        expected_files=("age_estimator.keras", "hair_classifier.keras", "gender_predictor.keras", "config.json"),
        file_types=("jpg", "jpeg", "png", "bmp"),
        input_label="Upload a single-face image",
        description="Predicts gender, confidence, estimated age, and age-route.",
    ),
    "senior": SpecConfig(
        key="senior",
        title="Senior Citizen",
        zip_name="senior_citizen_models.zip",
        expected_files=("senior_age_estimator.keras", "senior_gender_predictor.keras", "yolov8n.pt", "senior_config.json"),
        file_types=("jpg", "jpeg", "png", "bmp"),
        input_label="Upload an image frame with visible people",
        description="Detects people and marks age, gender, and senior-citizen status.",
    ),
    "sign": SpecConfig(
        key="sign",
        title="Sign Language",
        zip_name="sign_language_models.zip",
        expected_files=("sign_language_cnn.keras", "sign_language_config.json"),
        file_types=("jpg", "jpeg", "png"),
        input_label="Upload an ASL hand/sign image",
        description="Classifies the uploaded hand/sign image into an ASL class.",
    ),
    "car": SpecConfig(
        key="car",
        title="Car Colour",
        zip_name="car_colour_models.zip",
        expected_files=("colour_classifier.keras", "yolov8n.pt", "car_colour_config.json"),
        file_types=("jpg", "jpeg", "png"),
        input_label="Upload a traffic or car image",
        description="Detects cars and people, then predicts each car colour.",
    ),
    "nationality": SpecConfig(
        key="nationality",
        title="Nationality",
        zip_name="nationality_models.zip",
        expected_files=("nationality_detector.keras", "emotion_predictor.keras", "age_estimator.keras", "dress_colour_classifier.keras", "nationality_config.json"),
        file_types=("jpg", "jpeg", "png", "bmp"),
        input_label="Upload a single-face image",
        description="Predicts nationality and emotion, with conditional age and dress-colour output.",
    ),
    "voice": SpecConfig(
        key="voice",
        title="Voice Age Emotion",
        zip_name="voice_models.zip",
        expected_files=("voice_gender_classifier.keras", "voice_age_estimator.keras", "voice_emotion_detector.keras", "voice_config.json"),
        file_types=("wav", "mp3", "flac", "ogg"),
        input_label="Upload a voice note",
        description="Routes female voices, estimates male age, and predicts senior emotion when applicable.",
    ),
}


def spec_model_dir(spec: SpecConfig) -> Path:
    return MODEL_ROOT / spec.key


def expected_files_ready(spec: SpecConfig) -> bool:
    model_dir = spec_model_dir(spec)
    return all((model_dir / name).is_file() and (model_dir / name).stat().st_size > 0 for name in spec.expected_files)


def prepare_spec_models(spec: SpecConfig) -> None:
    """Extract only required model files from the spec zip into an isolated folder."""

    if expected_files_ready(spec):
        return

    model_dir = spec_model_dir(spec)
    model_dir.mkdir(parents=True, exist_ok=True)
    zip_path = PROJECT_ROOT / spec.zip_name
    if not zip_path.exists():
        return

    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(info.filename).name: info for info in zf.infolist() if Path(info.filename).name}
        for filename in spec.expected_files:
            if filename not in members:
                continue
            destination = model_dir / filename
            if destination.exists() and destination.stat().st_size > 0:
                continue
            with zf.open(members[filename]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def ensure_models(spec: SpecConfig) -> Path:
    with st.spinner("Preparing models..."):
        prepare_spec_models(spec)
    if not expected_files_ready(spec):
        missing = [name for name in spec.expected_files if not (spec_model_dir(spec) / name).exists()]
        st.error(
            "Models are not ready. Put the Kaggle output zip in the project root, then refresh this page.\n\n"
            f"Required zip: `{spec.zip_name}`\n\nMissing: {', '.join(missing)}"
        )
        st.stop()
    return spec_model_dir(spec)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def load_samples(spec_key: str) -> list[dict[str, Any]]:
    manifest = load_json(SAMPLE_MANIFEST)
    samples = manifest.get(spec_key, [])
    return [sample for sample in samples if (PROJECT_ROOT / sample.get("path", "")).exists()]


def save_upload(uploaded_file) -> Path:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower() or ".bin"
    fd, temp_path = tempfile.mkstemp(prefix="dashboard_", suffix=suffix, dir=str(UPLOAD_ROOT))
    os.close(fd)
    path = Path(temp_path)
    path.write_bytes(uploaded_file.getbuffer())
    return path


def image_details(path: Path) -> str:
    try:
        with Image.open(path) as image:
            return f"{image.width} x {image.height}, {path.stat().st_size / 1024:.1f} KB"
    except Exception:
        return f"{path.stat().st_size / 1024:.1f} KB"


def choose_input(spec: SpecConfig) -> Path | None:
    samples = load_samples(spec.key)
    options = ["Use sample", "Upload file"] if samples else ["Upload file"]
    mode = st.radio("Input", options, horizontal=True, label_visibility="collapsed")

    if mode == "Use sample":
        selected = st.selectbox(
            "Sample",
            samples,
            format_func=lambda item: item.get("name", Path(item["path"]).name),
        )
        path = PROJECT_ROOT / selected["path"]
        return path

    uploaded = st.file_uploader(spec.input_label, type=list(spec.file_types))
    if uploaded is None:
        return None
    return save_upload(uploaded)


def preview_input(path: Path, spec: SpecConfig) -> None:
    if spec.key == "voice":
        st.audio(str(path))
        st.caption(f"{path.name} - {path.stat().st_size / 1024:.1f} KB")
    else:
        st.image(str(path), caption=f"{path.name} - {image_details(path)}", use_container_width=True)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner=False)
def load_long_hair_engine(model_dir: str):
    from src.inference.inference_engine import InferenceEngine

    engine = InferenceEngine(model_dir=model_dir)
    engine.load_models()
    return engine


@st.cache_resource(show_spinner=False)
def load_senior_components(model_dir: str):
    from src.senior.age_estimator import SeniorAgeEstimator
    from src.senior.crop_preprocessor import CropPreprocessor
    from src.senior.frame_annotator import FrameAnnotator
    from src.senior.gender_predictor import SeniorGenderPredictor
    from src.senior.person_detector import PersonDetector
    from src.senior.senior_router import SeniorRouter

    model_path = Path(model_dir)
    detector = PersonDetector(str(model_path / "yolov8n.pt"))
    preprocessor = CropPreprocessor()
    age_estimator = SeniorAgeEstimator()
    age_estimator.load(str(model_path / "senior_age_estimator.keras"))
    gender_predictor = SeniorGenderPredictor()
    gender_predictor.load(str(model_path / "senior_gender_predictor.keras"))
    return detector, preprocessor, age_estimator, gender_predictor, SeniorRouter(), FrameAnnotator()


@st.cache_resource(show_spinner=False)
def load_sign_components(model_dir: str):
    from src.sign_language.classifier import DEFAULT_ASL_LABELS, SignLanguageClassifier
    from src.sign_language.preprocessor import SignLanguagePreprocessor

    config = load_json(Path(model_dir) / "sign_language_config.json")
    labels = config.get("class_labels") or DEFAULT_ASL_LABELS
    classifier = SignLanguageClassifier(str(Path(model_dir) / "sign_language_cnn.keras"), class_labels=labels)
    return classifier, SignLanguagePreprocessor()


@st.cache_resource(show_spinner=False)
def load_car_engine(model_dir: str):
    from src.inference.car_inference_engine import CarInferenceEngine

    engine = CarInferenceEngine(model_dir=model_dir)
    engine.load_models()
    return engine


@st.cache_resource(show_spinner=False)
def load_nationality_engine(model_dir: str):
    from src.nationality.inference_engine import NationalityInferenceEngine

    engine = NationalityInferenceEngine(model_dir=model_dir)
    engine.load_models()
    return engine


@st.cache_resource(show_spinner=False)
def load_voice_router(model_dir: str):
    from src.inference.voice_decision_router import VoiceDecisionRouter

    router = VoiceDecisionRouter(models_dir=model_dir)
    router.load_models()
    return router


def run_long_hair(path: Path, model_dir: Path) -> dict[str, Any]:
    result = load_long_hair_engine(str(model_dir)).predict(str(path))
    actual = {"label": result.label, "age_group": result.age_group}
    st.metric("Prediction", result.label)
    cols = st.columns(3)
    cols[0].metric("Confidence", f"{result.confidence * 100:.2f}%")
    cols[1].metric("Estimated age", result.estimated_age)
    cols[2].metric("Route", result.age_group)
    st.caption(result.age_group_display)
    return actual


def run_senior(path: Path, model_dir: Path) -> dict[str, Any]:
    detector, preprocessor, age_estimator, gender_predictor, router, annotator = load_senior_components(str(model_dir))
    frame = cv2.imread(str(path))
    detections = detector.detect(frame)
    rows = []
    annotated_pairs = []
    for detection in detections:
        crop = preprocessor.crop_and_preprocess(frame, detection.bbox)
        age, age_conf = age_estimator.predict(crop)
        gender, gender_conf = gender_predictor.predict(crop)
        decision = router.route(age, age_conf, gender, gender_conf)
        annotated_pairs.append((detection, decision))
        rows.append(
            {
                "Age": decision.display_age,
                "Gender": decision.display_gender,
                "Senior Citizen": "Yes" if decision.is_senior else "No",
                "Confidence": round(float(detection.confidence), 3),
            }
        )
    annotated = annotator.annotate(frame, annotated_pairs)
    senior_count = sum(1 for row in rows if row["Senior Citizen"] == "Yes")
    cols = st.columns(2)
    cols[0].metric("People detected", len(rows))
    cols[1].metric("Senior citizens", senior_count)
    st.image(bgr_to_rgb(annotated), caption="Output", use_container_width=True)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    return {"person_count": len(rows), "senior_count": senior_count}


def run_sign(path: Path, model_dir: Path) -> dict[str, Any]:
    classifier, preprocessor = load_sign_components(str(model_dir))
    frame = cv2.imread(str(path))
    processed = preprocessor.preprocess(frame)
    label, confidence = classifier.predict(processed)
    st.metric("Prediction", label)
    st.metric("Confidence", f"{confidence * 100:.2f}%")
    return {"label": label}


def run_car(path: Path, model_dir: Path) -> dict[str, Any]:
    result = load_car_engine(str(model_dir)).process_image(str(path))
    cols = st.columns(2)
    cols[0].metric("Cars detected", result.car_count)
    cols[1].metric("People detected", result.person_count)
    st.write("Car colours:", ", ".join(result.car_colours) if result.car_colours else "None")
    st.image(bgr_to_rgb(result.annotated_image), caption="Output", use_container_width=True)
    return {"car_count": result.car_count, "person_count": result.person_count, "car_colours": result.car_colours}


def run_nationality(path: Path, model_dir: Path) -> dict[str, Any]:
    result = load_nationality_engine(str(model_dir)).predict(str(path))
    cols = st.columns(2)
    cols[0].metric("Nationality", result.nationality)
    cols[1].metric("Emotion", result.emotion)
    st.caption(f"Nationality confidence: {result.nationality_confidence * 100:.2f}%")
    st.caption(f"Emotion confidence: {result.emotion_confidence * 100:.2f}%")
    if result.age is not None:
        st.metric("Age", result.age)
    if result.dress_colour is not None:
        st.metric("Dress colour", result.dress_colour)
    return {"nationality": result.nationality, "emotion": result.emotion}


def run_voice(path: Path, model_dir: Path) -> dict[str, Any]:
    result = load_voice_router(str(model_dir)).process(str(path))
    if result.error:
        st.error(result.message)
        return {"message": result.message}
    st.write(result.message)
    cols = st.columns(3)
    cols[0].metric("Gender route", result.gender or "-")
    cols[1].metric("Age", result.age if result.age is not None else "-")
    cols[2].metric("Senior", "Yes" if result.is_senior_citizen else "No")
    if result.emotion:
        st.metric("Emotion", result.emotion)
    return {
        "gender": result.gender,
        "senior": bool(result.is_senior_citizen),
        "emotion": result.emotion,
        "message": result.message,
    }


RUNNERS = {
    "long_hair": run_long_hair,
    "senior": run_senior,
    "sign": run_sign,
    "car": run_car,
    "nationality": run_nationality,
    "voice": run_voice,
}


def main() -> None:
    st.set_page_config(page_title="ML Projects Dashboard", layout="wide")

    with st.sidebar:
        st.title("ML Projects")
        selected = st.radio(
            "Spec",
            list(SPECS.keys()),
            format_func=lambda key: SPECS[key].title,
            label_visibility="collapsed",
        )

    spec = SPECS[selected]
    st.title(spec.title)
    st.caption(spec.description)

    model_dir = ensure_models(spec)
    path = choose_input(spec)
    if path is None:
        return

    left, right = st.columns([1, 1])
    with left:
        preview_input(path, spec)
    with right:
        if st.button("Run", type="primary", use_container_width=True):
            with st.spinner("Running..."):
                RUNNERS[spec.key](path, model_dir)


if __name__ == "__main__":
    main()
