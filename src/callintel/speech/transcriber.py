"""
faster-whisper speech transcription pipeline.
Transcribes audio files into timestamped turns with acoustic turn segmentation and speaker attribution.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import re
from src.callintel.speech.audio_utils import load_and_normalize_audio, format_timestamp
from src.callintel.speech.diarizer import AcousticDiarizer


class WhisperTranscriber:
    """Production ASR service wrapping faster-whisper with CTranslate2."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.diarizer = AcousticDiarizer(target_sample_rate=16000)

    def _ensure_loaded(self):
        if self.model is None:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes an audio file into timestamped sentence turns with acoustic voice-based speaker diarization.
        """
        self._ensure_loaded()
        assert self.model is not None, "Whisper model failed to load"
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio_array, sr, duration_sec = load_and_normalize_audio(path)

        segments, info = self.model.transcribe(
            audio_array,
            beam_size=beam_size,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
        )

        candidate_turns = []
        for segment in segments:
            seg_text = segment.text.strip()
            if not seg_text:
                continue

            start_s = float(segment.start)
            end_s = float(segment.end)
            candidate_turns.append({
                "start_seconds": round(start_s, 3),
                "end_seconds": round(end_s, 3),
                "start_time": format_timestamp(start_s),
                "end_time": format_timestamp(end_s),
                "utterance": seg_text,
                "avg_logprob": round(float(segment.avg_logprob), 3),
            })

        # Apply acoustic speaker diarization on audio array and candidate turns
        turns = self.diarizer.diarize_segments(
            audio_array=audio_array,
            segments=candidate_turns,
            expected_speakers=2,
        )

        return {
            "audio_path": str(path),
            "duration_seconds": round(duration_sec, 2),
            "language": info.language,
            "language_probability": round(float(info.language_probability), 3),
            "total_turns": len(turns),
            "speakers_detected": sorted(list({t["speaker_cluster_id"] for t in turns})),
            "full_transcript": " ".join(t.get("utterance", "") for t in turns),
            "turns": turns,
        }

    transcribe = transcribe_file
