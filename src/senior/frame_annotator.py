"""Frame annotation module for the Senior Citizen Identification system.

Draws color-coded bounding boxes and text labels on video frames
based on detection and classification results.
"""

from typing import List, Tuple

import cv2
import numpy as np

from src.senior.models import Detection, ClassificationResult


# Type alias for annotated detections passed to the annotator
AnnotatedDetection = Tuple[Detection, ClassificationResult]


class FrameAnnotator:
    """Draws bounding boxes and labels on video frames.

    Annotates each detected person with a color-coded bounding box
    and a text label positioned above the top edge of the box.

    Colors:
        - Green (0, 255, 0): Senior Citizens (age > 60, confidence >= 0.3)
        - Blue (255, 0, 0): Non-Seniors (age <= 60, confidence >= 0.3)
        - Yellow (0, 255, 255): Low-confidence estimates (confidence < 0.3)
    """

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    BOX_THICKNESS = 2
    LABEL_OFFSET_Y = 10  # Pixels above the top edge of the bounding box

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[AnnotatedDetection],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame.

        Args:
            frame: Input BGR image as a numpy array (H x W x 3).
            detections: List of (Detection, ClassificationResult) tuples
                containing the bounding box location and classification info.

        Returns:
            A new numpy array with annotations drawn, same shape as input.
        """
        annotated = frame.copy()

        for detection, classification in detections:
            bbox = detection.bbox
            color = classification.box_color
            label = classification.label_text

            # Draw bounding box rectangle
            top_left = (bbox.x, bbox.y)
            bottom_right = (bbox.x + bbox.width, bbox.y + bbox.height)
            cv2.rectangle(annotated, top_left, bottom_right, color, self.BOX_THICKNESS)

            # Position label above the top edge of the bounding box
            label_y = max(bbox.y - self.LABEL_OFFSET_Y, 15)
            label_position = (bbox.x, label_y)
            cv2.putText(
                annotated,
                label,
                label_position,
                self.FONT,
                self.FONT_SCALE,
                color,
                self.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated
