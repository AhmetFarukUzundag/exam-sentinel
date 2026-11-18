"""src/vision.py - Vision processing pipeline using YOLOv8n and MediaPipe Face Landmarker.

Detects candidate absences, multiple people, cell phones, and gaze deviation (head pose)
across video frames. Also tracks facial bounding boxes for privacy anonymization.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

from src.schemas import BoundingBox, ViolationType

logger = logging.getLogger(__name__)

@dataclass
class FrameAnalysisResult:
    """Detection results for a single processed video frame."""

    frame_idx: int
    timestamp_sec: float
    violations: List[ViolationType] = field(default_factory=list)
    face_boxes: List[BoundingBox] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)


class HeadPoseEstimator:
    """Estimates head orientation (Yaw, Pitch, Roll) using 2D landmarks and 3D reference model."""

    # 3D canonical human face model coordinates in arbitrary world units
    MODEL_POINTS_3D = np.array(
        [
            (0.0, 0.0, 0.0),          # 1: Nose tip
            (0.0, -330.0, -65.0),     # 152: Chin
            (-225.0, 170.0, -135.0),  # 33: Left eye outer corner
            (225.0, 170.0, -135.0),   # 263: Right eye outer corner
            (-150.0, -150.0, -125.0), # 61: Left mouth corner
            (150.0, -150.0, -125.0),  # 291: Right mouth corner
        ],
        dtype=np.float64,
    )

    # Corresponding MediaPipe landmark indices
    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def estimate_pose(
        self,
        landmarks: List[object],
        img_w: int,
        img_h: int,
    ) -> Optional[Tuple[float, float, float]]:
        """Calculate Yaw, Pitch, Roll angles in degrees.

        Args:
            landmarks: Normalized landmark list from MediaPipe for a single face.
            img_w: Frame width in pixels.
            img_h: Frame height in pixels.

        Returns:
            Tuple of (pitch, yaw, roll) in degrees, or None if calculation fails.
        """
        image_points_2d = np.array(
            [
                (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
                for idx in self.LANDMARK_INDICES
            ],
            dtype=np.float64,
        )

        # Approximate camera intrinsic parameters
        focal_length = float(img_w)
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _ = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Decompose rotation matrix into Euler angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        pitch, yaw, roll = angles[0], angles[1], angles[2]

        return float(pitch), float(yaw), float(roll)


class VisionAnalyzer:
    """Processes video frames through YOLOv8 and MediaPipe Face Landmarker."""

    # YOLO class IDs (COCO dataset)
    CLASS_PERSON = 0
    CLASS_CELL_PHONE = 67

    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        face_task_path: str = "models/face_landmarker.task",
        yaw_threshold: float = 30.0,
        pitch_threshold: float = 25.0,
        yolo_conf: float = 0.35,
    ) -> None:
        """Initialize models and detection thresholds.

        Args:
            yolo_model_path: Path to YOLO weights.
            face_task_path: Path to MediaPipe task asset.
            yaw_threshold: Maximum allowed head yaw angle (degrees) before gaze violation.
            pitch_threshold: Maximum allowed head pitch angle (degrees) before gaze violation.
            yolo_conf: Confidence threshold for YOLO detections.
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.yolo_conf = yolo_conf

        logger.info("Initializing YOLO model from %s...", yolo_model_path)
        self.yolo = YOLO(yolo_model_path)

        logger.info("Initializing MediaPipe Face Landmarker from %s...", face_task_path)
        base_options = mp_python.BaseOptions(model_asset_path=face_task_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.pose_estimator = HeadPoseEstimator()

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
    ) -> FrameAnalysisResult:
        """Analyze a single video frame for rule violations.

        Args:
            frame_bgr: Raw video frame matrix in BGR format.
            frame_idx: Index of current frame.
            timestamp_sec: Timestamp of current frame in seconds.

        Returns:
            FrameAnalysisResult containing detected violations and bounding boxes.
        """
        h, w = frame_bgr.shape[:2]
        result = FrameAnalysisResult(frame_idx=frame_idx, timestamp_sec=timestamp_sec)

        yolo_results = self.yolo(
            frame_bgr,
            classes=[self.CLASS_PERSON, self.CLASS_CELL_PHONE],
            conf=self.yolo_conf,
            verbose=False,
        )

        person_count = 0
        phone_count = 0

        for box in yolo_results[0].boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == self.CLASS_PERSON:
                person_count += 1
            elif cls_id == self.CLASS_CELL_PHONE:
                phone_count += 1

        result.details["person_count"] = person_count
        result.details["phone_count"] = phone_count

        if phone_count > 0:
            result.violations.append(ViolationType.CELL_PHONE)

        if person_count > 1:
            result.violations.append(ViolationType.MULTIPLE_FACES)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        face_result = self.landmarker.detect(mp_image)

        face_count = len(face_result.face_landmarks)
        result.details["mediapipe_face_count"] = face_count

        if face_count == 0:
            # Candidate absence detected
            result.violations.append(ViolationType.ABSENCE)
        elif face_count > 1 and ViolationType.MULTIPLE_FACES not in result.violations:
            result.violations.append(ViolationType.MULTIPLE_FACES)

        # Extract facial bounding boxes and head pose for primary face
        for landmarks in face_result.face_landmarks:
            # Compute face bounding box from min/max landmark coordinates
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            bbox = BoundingBox(
                xmin=max(0, int(min(xs))),
                ymin=max(0, int(min(ys))),
                xmax=min(w, int(max(xs))),
                ymax=min(h, int(max(ys))),
            )
            result.face_boxes.append(bbox)

        # Evaluate head pose on the first (primary) detected face
        if face_count > 0:
            primary_landmarks = face_result.face_landmarks[0]
            angles = self.pose_estimator.estimate_pose(primary_landmarks, w, h)
            if angles is not None:
                pitch, yaw, roll = angles
                result.details["pitch"] = round(pitch, 1)
                result.details["yaw"] = round(yaw, 1)
                result.details["roll"] = round(roll, 1)

                if abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold:
                    result.violations.append(ViolationType.GAZE_AWAY)

        return result

    def process_video(
        self,
        video_path: str,
        frame_stride: int = 5,
    ) -> Tuple[List[FrameAnalysisResult], float, int]:
        """Iterate through video frames at specified stride and collect frame results.

        Args:
            video_path: Path to target .mp4 video file.
            frame_stride: Process every N-th frame to optimize throughput.

        Returns:
            Tuple of (frame_results_list, video_duration_seconds, total_frame_count).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        logger.info(
            "Processing video %s (FPS: %.2f, Total Frames: %d, Duration: %.2fs, Stride: %d)...",
            video_path,
            fps,
            total_frames,
            duration_sec,
            frame_stride,
        )

        results: List[FrameAnalysisResult] = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_stride == 0:
                    timestamp_sec = round(frame_idx / fps, 3)
                    frame_result = self.analyze_frame(frame, frame_idx, timestamp_sec)
                    results.append(frame_result)

                frame_idx += 1
        finally:
            cap.release()

        return results, duration_sec, total_frames
