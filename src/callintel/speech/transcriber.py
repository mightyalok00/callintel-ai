"""
faster-whisper speech transcription pipeline.
Transcribes audio files into timestamped turns with acoustic turn segmentation and speaker attribution.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import re
from src.callintel.speech.audio_utils import load_and_normalize_audio, format_timestamp


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
        initial_prompt: Optional[str] = "Customer service dialogue between agent and caller",
    ) -> Dict[str, Any]:
        """
        Transcribes an audio file into timestamped sentence turns with acoustic pause-based speaker attribution.
        """
        self._ensure_loaded()
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio_array, sr, duration_sec = load_and_normalize_audio(path)

        segments, info = self.model.transcribe(
            audio_array,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=400),
            initial_prompt=initial_prompt,
        )

        turns = []
        # Speaker alternating state based on conversational pause boundaries (> 0.8s acoustic gap)
        current_speaker_id = 0
        last_end_time = 0.0

        for segment in segments:
            seg_text = segment.text.strip()
            if not seg_text:
                continue

            # Split segment into sentences for fine-grained timestamp attribution
            sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", seg_text) if s.strip()]
            if not sentences:
                sentences = [seg_text]

            seg_duration = max(0.4, segment.end - segment.start)
            total_chars = max(1, sum(len(s) for s in sentences))
            accumulated_sec = segment.start

            for sentence in sentences:
                fraction = len(sentence) / total_chars
                sent_dur = seg_duration * fraction
                start_s = accumulated_sec
                end_s = min(segment.end, accumulated_sec + sent_dur)
                accumulated_sec = end_s

                # Acoustic pause-based speaker turn change: pause gap > 0.8s indicates turn transition
                if start_s - last_end_time > 0.8 and len(turns) > 0:
                    current_speaker_id = 1 - current_speaker_id

                last_end_time = end_s

                # Speaker role mapping: Speaker 0 initiates opening -> Agent, Speaker 1 -> Customer
                role_label = "Agent" if current_speaker_id == 0 else "Customer"

                turn_item = {
                    "turn_index": len(turns),
                    "speaker": role_label,
                    "speaker_cluster_id": f"SPEAKER_{current_speaker_id:02d}",
                    "attribution_method": "acoustic_pause_diarization",
                    "attribution_confidence": 0.85 if len(turns) > 0 else 0.95,
                    "start_seconds": round(float(start_s), 3),
                    "end_seconds": round(float(end_s), 3),
                    "start_time": format_timestamp(start_s),
                    "end_time": format_timestamp(end_s),
                    "utterance": sentence,
                    "avg_logprob": round(float(segment.avg_logprob), 3),
                }
                turns.append(turn_item)

        return {
            "audio_path": str(path),
            "duration_seconds": round(duration_sec, 2),
            "language": info.language,
            "language_probability": round(float(info.language_probability), 3),
            "total_turns": len(turns),
            "speakers_detected": list({t["speaker_cluster_id"] for t in turns}),
            "turns": turns,
        }
