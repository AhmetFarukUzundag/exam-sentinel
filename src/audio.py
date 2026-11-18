"""src/audio.py - Audio processing and Voice Activity Detection (VAD) module.

Extracts audio stream directly from video files via PyAV without external FFmpeg CLI,
resamples to 16kHz mono, and detects human speech segments using Silero VAD (ONNX runtime).
"""

import logging
from pathlib import Path
from typing import List, Optional

import av
import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from src.schemas import ViolationEvent, ViolationType

logger = logging.getLogger(__name__)


class AudioExtractionError(Exception):
    """Raised when audio stream extraction fails."""
    pass


class AudioProcessor:
    """Handles audio extraction and resampling from video containers using PyAV."""

    def __init__(self, target_sample_rate: int = 16000) -> None:
        """Initialize AudioProcessor.

        Args:
            target_sample_rate: Expected audio sample rate in Hz (default 16000 for Silero VAD).
        """
        self.target_sample_rate: int = target_sample_rate

    def extract_audio_tensor(self, video_path: str) -> Optional[torch.Tensor]:
        """Extract and resample audio stream to a 1D PyTorch float32 tensor.

        Args:
            video_path: Path to the input video (.mp4) file.

        Returns:
            1D torch.FloatTensor of shape (samples,) normalized to [-1.0, 1.0],
            or None if the video contains no audio stream.

        Raises:
            AudioExtractionError: If the container cannot be opened or decoded.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            container = av.open(str(path))
        except Exception as e:
            raise AudioExtractionError(f"Failed to open video container: {e}") from e

        # Check if audio stream exists
        if not container.streams.audio:
            logger.warning("No audio stream found in video: %s", video_path)
            container.close()
            return None

        audio_stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="flt",
            layout="mono",
            rate=self.target_sample_rate
        )

        audio_chunks: List[np.ndarray] = []

        try:
            for frame in container.decode(audio_stream):
                resampled_frames = resampler.resample(frame)
                for r_frame in resampled_frames:
                    # r_frame.to_ndarray() has shape (1, num_samples)
                    chunk = r_frame.to_ndarray()[0]
                    audio_chunks.append(chunk)

            # Flush any remaining audio in the resampler buffer
            flush_frames = resampler.resample(None)
            if flush_frames:
                for r_frame in flush_frames:
                    chunk = r_frame.to_ndarray()[0]
                    audio_chunks.append(chunk)

        except Exception as e:
            raise AudioExtractionError(f"Error while decoding audio stream: {e}") from e
        finally:
            container.close()

        if not audio_chunks:
            logger.warning("Audio stream present but contained no decodable frames: %s", video_path)
            return None

        # Concatenate all chunks into a continuous 1D array
        full_audio = np.concatenate(audio_chunks, axis=0).astype(np.float32)

        # Ensure amplitude stays within [-1.0, 1.0]
        full_audio = np.clip(full_audio, -1.0, 1.0)

        return torch.from_numpy(full_audio)


class SpeechDetector:
    """Detects human vocal activity in exam videos using Silero VAD."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 200,
        sampling_rate: int = 16000,
    ) -> None:
        """Initialize SpeechDetector with Silero VAD ONNX model.

        Args:
            threshold: Speech probability threshold (0.0 to 1.0) above which speech is triggered.
            min_speech_duration_ms: Minimum speech segment duration in ms to avoid false spikes.
            min_silence_duration_ms: Minimum silence duration in ms to separate two speech segments.
            sampling_rate: Audio sampling rate in Hz (must be 16000 or 8000).
        """
        self.threshold: float = threshold
        self.min_speech_duration_ms: int = min_speech_duration_ms
        self.min_silence_duration_ms: int = min_silence_duration_ms
        self.sampling_rate: int = sampling_rate

        # Load Silero VAD using ONNX backend for cross-platform stability (handles non-ASCII paths)
        logger.info("Loading Silero VAD model (ONNX backend)...")
        self.model = load_silero_vad(onnx=True)
        self.audio_processor = AudioProcessor(target_sample_rate=sampling_rate)

    def detect(self, video_path: str) -> List[ViolationEvent]:
        """Process video audio and return a list of speech violation events.

        Args:
            video_path: Path to the target video (.mp4).

        Returns:
            List of ViolationEvent instances representing speech violation intervals.
        """
        audio_tensor = self.audio_processor.extract_audio_tensor(video_path)
        if audio_tensor is None or audio_tensor.numel() == 0:
            return []

        # get_speech_timestamps returns list of dicts: [{'start': int_sample, 'end': int_sample}]
        raw_timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=False,
        )

        violations: List[ViolationEvent] = []

        for item in raw_timestamps:
            # Convert sample indices to precise timestamps in seconds
            start_sec = round(float(item["start"]) / self.sampling_rate, 3)
            end_sec = round(float(item["end"]) / self.sampling_rate, 3)

            event = ViolationEvent(
                violation_type=ViolationType.SPEECH,
                start_time=start_sec,
                end_time=end_sec,
                confidence=self.threshold,
                details={
                    "start_sample": item["start"],
                    "end_sample": item["end"],
                    "duration_sec": round(end_sec - start_sec, 3),
                },
            )
            violations.append(event)

        logger.info(
            "Speech detection finished for %s: found %d speech violation(s).",
            video_path,
            len(violations),
        )
        return violations
