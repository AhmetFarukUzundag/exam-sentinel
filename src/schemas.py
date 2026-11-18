"""src/schemas.py - Exam Sentinel Data Contracts.

This module strictly defines the data structures and validation schemas
used throughout the entire pipeline (audio, vision, postprocessing, reporting).
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ViolationType(str, Enum):
    """Enumeration of violation categories detectable during an exam."""

    ABSENCE = "absence"                  # Candidate is absent from the frame
    MULTIPLE_FACES = "multiple_faces"    # Multiple individuals detected in frame
    CELL_PHONE = "cell_phone"            # Cell phone detected in frame
    SPEECH = "speech"                    # Speech or vocal activity detected
    GAZE_AWAY = "gaze_away"              # Gaze deviation or excessive head rotation


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected objects or faces."""

    model_config = ConfigDict(frozen=True)

    xmin: int = Field(..., ge=0, description="Left horizontal coordinate")
    ymin: int = Field(..., ge=0, description="Top vertical coordinate")
    xmax: int = Field(..., ge=0, description="Right horizontal coordinate")
    ymax: int = Field(..., ge=0, description="Bottom vertical coordinate")

    @property
    def width(self) -> int:
        """Returns the width of the bounding box."""
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        """Returns the height of the bounding box."""
        return self.ymax - self.ymin


class ViolationEvent(BaseModel):
    """A discrete violation event on the timeline.
    
    Example:
        Candidate interacting with a phone between 00:14 and 00:18.
    """

    violation_type: ViolationType = Field(
        ..., description="Category of the violation"
    )
    start_time: float = Field(
        ..., ge=0.0, description="Timestamp (seconds) when violation started"
    )
    end_time: float = Field(
        ..., ge=0.0, description="Timestamp (seconds) when violation ended"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model prediction confidence score (0.0 to 1.0)"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata details (e.g., yaw angle degree, detected person count)"
    )

    @property
    def duration(self) -> float:
        """Returns total duration of the event in seconds."""
        return max(0.0, self.end_time - self.start_time)


class EvaluationMetrics(BaseModel):
    """Model evaluation metrics against ground truth annotations."""

    precision: float = Field(..., ge=0.0, le=1.0, description="Precision: TP / (TP + FP)")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall: TP / (TP + FN)")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="Harmonic mean of precision and recall")
    true_positives: int = Field(..., ge=0, description="Correctly identified violation count")
    false_positives: int = Field(..., ge=0, description="False alarm count (predicted as violation but wasn't)")
    false_negatives: int = Field(..., ge=0, description="Missed violation count (actual violation not caught)")


class ExamReport(BaseModel):
    """Final examination surveillance audit report."""

    video_path: str = Field(..., description="File path of the audited video")
    duration_seconds: float = Field(..., ge=0.0, description="Total video duration in seconds")
    total_violations: int = Field(default=0, ge=0, description="Total count of distinct violation events")
    violations_by_type: Dict[ViolationType, int] = Field(
        default_factory=dict,
        description="Violation count aggregated by category"
    )
    events: List[ViolationEvent] = Field(
        default_factory=list,
        description="Chronologically sorted list of violation events"
    )
    evaluation: Optional[EvaluationMetrics] = Field(
        default=None,
        description="Optional evaluation metrics if ground truth was supplied"
    )

