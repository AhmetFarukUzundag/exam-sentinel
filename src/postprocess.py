"""src/postprocess.py - Event debouncing, noise filtering, and temporal smoothing.

Converts frame-level instantaneous violation signals into contiguous time intervals,
filters out transient false-positive spikes, and merges closely spaced events.
"""

import logging
from collections import defaultdict
from typing import Dict, List

from src.schemas import ViolationEvent, ViolationType
from src.vision import FrameAnalysisResult

logger = logging.getLogger(__name__)


class EventPostProcessor:
    """Filters noisy frame detections and consolidates them into discrete violation intervals."""

    def __init__(
        self,
        min_duration_sec: float = 1.0,
        merge_gap_sec: float = 1.5,
    ) -> None:
        """Initialize EventPostProcessor with temporal thresholds.

        Args:
            min_duration_sec: Minimum continuous event duration to qualify as a valid violation.
                             Shorter spikes are discarded as transient noise (false alarms).
            merge_gap_sec: Maximum time gap between two occurrences of the same violation type
                           to merge them into a single continuous event.
        """
        self.min_duration_sec: float = min_duration_sec
        self.merge_gap_sec: float = merge_gap_sec

    def _cluster_frame_violations(
        self,
        frame_results: List[FrameAnalysisResult],
    ) -> Dict[ViolationType, List[ViolationEvent]]:
        """Group consecutive frame detections of the same category into initial time blocks.

        Args:
            frame_results: Chronologically ordered frame analysis results from VisionAnalyzer.

        Returns:
            Dictionary mapping each ViolationType to raw unmerged ViolationEvent intervals.
        """
        # Map: violation_type -> list of timestamps where it appeared
        type_timestamps: Dict[ViolationType, List[float]] = defaultdict(list)

        for fr in frame_results:
            for v_type in fr.violations:
                type_timestamps[v_type].append(fr.timestamp_sec)

        clustered_events: Dict[ViolationType, List[ViolationEvent]] = defaultdict(list)

        for v_type, timestamps in type_timestamps.items():
            if not timestamps:
                continue

            # Sort defensively to ensure chronological order
            timestamps.sort()

            current_start = timestamps[0]
            current_end = timestamps[0]

            for ts in timestamps[1:]:
                # If timestamp is close to the previous frame timestamp (within merge gap)
                if ts - current_end <= self.merge_gap_sec:
                    current_end = ts
                else:
                    # Finalize current block
                    clustered_events[v_type].append(
                        ViolationEvent(
                            violation_type=v_type,
                            start_time=round(current_start, 3),
                            end_time=round(current_end, 3),
                            confidence=0.85,
                            details={"raw_frame_count": round((current_end - current_start), 2)},
                        )
                    )
                    current_start = ts
                    current_end = ts

            # Append the trailing block
            clustered_events[v_type].append(
                ViolationEvent(
                    violation_type=v_type,
                    start_time=round(current_start, 3),
                    end_time=round(current_end, 3),
                    confidence=0.85,
                    details={"raw_frame_count": round((current_end - current_start), 2)},
                )
            )

        return clustered_events

    def _filter_and_merge(
        self,
        events: List[ViolationEvent],
    ) -> List[ViolationEvent]:
        """Apply duration threshold filtering and merge adjacent events.

        Args:
            events: Raw event list of a specific violation category.

        Returns:
            Cleaned and merged list of ViolationEvent instances.
        """
        if not events:
            return []

        # 1. Filter out short transient spikes (Duration < min_duration_sec)
        valid_events = [ev for ev in events if ev.duration >= self.min_duration_sec]

        if not valid_events:
            return []

        # 2. Merge adjacent events separated by less than merge_gap_sec
        merged: List[ViolationEvent] = [valid_events[0]]

        for current in valid_events[1:]:
            last_event = merged[-1]

            gap = current.start_time - last_event.end_time
            if gap <= self.merge_gap_sec:
                # Merge into previous event by extending its end_time
                new_confidence = max(last_event.confidence, current.confidence)
                merged[-1] = ViolationEvent(
                    violation_type=last_event.violation_type,
                    start_time=last_event.start_time,
                    end_time=max(last_event.end_time, current.end_time),
                    confidence=new_confidence,
                    details={
                        "merged": True,
                        "original_intervals": [
                            [last_event.start_time, last_event.end_time],
                            [current.start_time, current.end_time],
                        ],
                    },
                )
            else:
                merged.append(current)

        return merged

    def process(
        self,
        frame_results: List[FrameAnalysisResult],
        audio_events: List[ViolationEvent],
    ) -> List[ViolationEvent]:
        """Consolidate vision and audio violation events into a unified, clean timeline.

        Args:
            frame_results: Raw frame-by-frame vision analysis results.
            audio_events: Discrete speech violation events already extracted by audio processor.

        Returns:
            Complete, chronologically ordered list of confirmed, noise-free violation events.
        """
        all_cleaned_events: List[ViolationEvent] = []

        # 1. Process Vision Violations
        clustered_vision = self._cluster_frame_violations(frame_results)
        for v_type, raw_events in clustered_vision.items():
            cleaned = self._filter_and_merge(raw_events)
            all_cleaned_events.extend(cleaned)

        # 2. Process Audio Violations (Apply duration filter & merge on speech events too)
        cleaned_audio = self._filter_and_merge(audio_events)
        all_cleaned_events.extend(cleaned_audio)

        # 3. Sort all events chronologically by start_time
        all_cleaned_events.sort(key=lambda ev: ev.start_time)

        logger.info(
            "Postprocessing complete: %d clean violation event(s) generated.",
            len(all_cleaned_events),
        )
        return all_cleaned_events
