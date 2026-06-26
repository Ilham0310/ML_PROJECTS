"""Property tests for Car Colour Detection."""

from unittest.mock import MagicMock

import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

from src.annotation.annotation_engine import AnnotationEngine
from src.car_colour.models import BoundingBox, Detection
from src.classification.colour_classifier import ColourClassifier
from src.utils.count_utils import count_by_class
from src.utils.image_utils import scale_to_fit


def _box_strategy():
    return st.builds(
        BoundingBox,
        x1=st.integers(0, 200),
        y1=st.integers(0, 200),
        x2=st.integers(201, 500),
        y2=st.integers(201, 500),
        confidence=st.floats(0, 1, allow_nan=False, allow_infinity=False),
        class_id=st.integers(0, 5),
    )


def _detection_strategy():
    return st.builds(
        Detection,
        bbox=_box_strategy(),
        class_id=st.integers(0, 5),
        colour_label=st.none(),
    )


# Feature: car-colour-detection, Property 1: Classifier single-label output
@given(
    image=arrays(
        np.uint8,
        st.tuples(
            st.integers(1, 96),
            st.integers(1, 96),
            st.just(3),
        ),
    ),
    index=st.integers(0, 7),
)
@settings(max_examples=100, deadline=None)
def test_colour_classifier_single_label_output_property(image, index):
    model = MagicMock()
    probabilities = np.zeros((1, 8), dtype=np.float32)
    probabilities[0, index] = 1.0
    model.predict.return_value = probabilities
    classifier = ColourClassifier(model=model)
    processed = classifier.preprocess(image)
    label = classifier.classify(image)
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert 0.0 <= float(processed.min()) <= float(processed.max()) <= 1.0
    assert label in ColourClassifier.COLOUR_LABELS


# Feature: car-colour-detection, Property 2: Annotation colour mapping
@given(label=st.sampled_from(ColourClassifier.COLOUR_LABELS))
@settings(max_examples=100)
def test_annotation_colour_mapping_property(label):
    engine = AnnotationEngine()
    expected = (0, 0, 255) if label == "blue" else (255, 0, 0)
    assert engine.get_rectangle_colour(label) == expected


# Feature: car-colour-detection, Property 3: Car count accuracy
@given(detections=st.lists(_detection_strategy(), min_size=0, max_size=100))
@settings(max_examples=100)
def test_car_count_accuracy_property(detections):
    assert count_by_class(detections, 2) == sum(1 for item in detections if item.class_id == 2)


# Feature: car-colour-detection, Property 4: Person count accuracy
@given(detections=st.lists(_detection_strategy(), min_size=0, max_size=100))
@settings(max_examples=100)
def test_person_count_accuracy_property(detections):
    assert count_by_class(detections, 0) == sum(1 for item in detections if item.class_id == 0)


# Feature: car-colour-detection, Property 5: Image scaling preserves aspect ratio
@given(
    image_width=st.integers(1, 5000),
    image_height=st.integers(1, 5000),
    max_width=st.integers(1, 1000),
    max_height=st.integers(1, 1000),
)
@settings(max_examples=100)
def test_image_scaling_preserves_aspect_ratio_property(
    image_width,
    image_height,
    max_width,
    max_height,
):
    scaled_width, scaled_height = scale_to_fit(
        image_width,
        image_height,
        max_width,
        max_height,
    )
    assert scaled_width <= max_width
    assert scaled_height <= max_height
    assert scaled_width >= 1
    assert scaled_height >= 1
    ideal_scale = min(max_width / image_width, max_height / image_height)
    ideal_width = min(max_width, max(1, int(round(image_width * ideal_scale))))
    ideal_height = min(max_height, max(1, int(round(image_height * ideal_scale))))
    assert scaled_width == ideal_width
    assert scaled_height == ideal_height


# Feature: car-colour-detection, Property 6: Classification failure defaults to blue rectangle
@given(label=st.one_of(st.none(), st.just(""), st.text(min_size=1).filter(lambda value: value.lower() != "blue")))
@settings(max_examples=100)
def test_classification_failure_defaults_to_blue_rectangle_property(label):
    assert AnnotationEngine().get_rectangle_colour(label) == (255, 0, 0)
