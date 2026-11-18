"""src/anonymizer.py - Video anonymization module for KVKK/GDPR compliance.

Applies Gaussian Blur to detected facial bounding boxes and exports
the processed frames as a new anonymous MP4 surveillance video.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.schemas import BoundingBox
from src.vision import FrameAnalysisResult

logger = logging.getLogger(__name__)


class VideoAnonymizer:
    """Anonymizes human faces in video frames using OpenCV Gaussian Blur."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (51, 51),
        sigma_x: float = 30.0,
    ) -> None:
        """Initialize VideoAnonymizer.

        Args:
            kernel_size: Gaussian kernel dimensions (must be odd positive integers).
            sigma_x: Gaussian kernel standard deviation in X direction. Higher values = stronger blur.
        """
        # Ensure kernel dimensions are odd numbers
        kx = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
        ky = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1
        self.kernel_size: Tuple[int, int] = (kx, ky)
        self.sigma_x: float = sigma_x

    def anonymize_frame(
        self,
        frame: np.ndarray,
        face_boxes: List[BoundingBox],
    ) -> np.ndarray:
        """Apply Gaussian Blur to facial regions of interest (ROI) within a single frame.

        Args:
            frame: Raw BGR video frame array.
            face_boxes: List of BoundingBox instances defining detected face coordinates.

        Returns:
            New frame array with blurred facial regions.
        """
        if not face_boxes:
            return frame

        processed = frame.copy()
        h, w = processed.shape[:2]

        for box in face_boxes:
            # Clamp coordinates defensively within frame boundaries
            x1 = max(0, min(w - 1, box.xmin))
            y1 = max(0, min(h - 1, box.ymin))
            x2 = max(0, min(w, box.xmax))
            y2 = max(0, min(h, box.ymax))

            # Skip invalid or empty bounding boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Extract face Region of Interest (ROI)
            face_roi = processed[y1:y2, x1:x2]

            # Adjust kernel size if face ROI is smaller than the default kernel
            roi_h, roi_w = face_roi.shape[:2]
            kx = min(self.kernel_size[0], roi_w if roi_w % 2 != 0 else roi_w - 1)
            ky = min(self.kernel_size[1], roi_h if roi_h % 2 != 0 else roi_h - 1)

            # Kernel must be at least 3x3
            if kx < 3 or ky < 3:
                kx, ky = 3, 3

            # Apply Gaussian Blur
            blurred_roi = cv2.GaussianBlur(face_roi, (kx, ky), sigmaX=self.sigma_x)

            # Splice blurred ROI back into the frame
            processed[y1:y2, x1:x2] = blurred_roi

            # Optional: Draw subtle privacy boundary indicator
            cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 255), 1)

        return processed

    def anonymize_video(
        self,
        input_video_path: str,
        output_video_path: str,
        frame_results: List[FrameAnalysisResult],
    ) -> str:
        """Read input video, blur faces using cached detection results, and save anonymous video.

        Args:
            input_video_path: File path of input exam video.
            output_video_path: Destination path for blurred MP4 video.
            frame_results: Frame analysis results from VisionAnalyzer containing face bounding boxes.

        Returns:
            Resolved string path of the exported anonymized video.
        """
        in_path = Path(input_video_path)
        if not in_path.is_file():
            raise FileNotFoundError(f"Source video not found: {input_video_path}")

        out_path = Path(output_video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build lookup table: frame_idx -> List[BoundingBox]
        face_box_map: Dict[int, List[BoundingBox]] = {
            fr.frame_idx: fr.face_boxes for fr in frame_results if fr.face_boxes
        }

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use standard MP4V codec for maximum cross-platform compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        logger.info(
            "Anonymizing video %s -> %s (%dx%d @ %.1f FPS)...",
            input_video_path,
            output_video_path,
            width,
            height,
            fps,
        )

        frame_idx = 0
        last_known_boxes: List[BoundingBox] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # If this frame had a direct detection, update last known boxes
                if frame_idx in face_box_map:
                    last_known_boxes = face_box_map[frame_idx]

                # Blur using the latest known bounding boxes (handles frame-stride gaps)
                anonymized = self.anonymize_frame(frame, last_known_boxes)
                writer.write(anonymized)

                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        logger.info("Anonymized video exported successfully to: %s", str(out_path))
        return str(out_path)
