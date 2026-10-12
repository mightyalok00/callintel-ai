"""
Audio processing utilities for Call Intelligence Engine.
Handles audio normalization (16kHz, mono, 16-bit PCM WAV).
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import soundfile as sf


def load_and_normalize_audio(
    file_path: Union[str, Path],
    target_sample_rate: int = 16000,
) -> Tuple[np.ndarray, int, float]:
    """
    Loads an audio file, converts to mono, resamples to target_sample_rate if needed,
    and normalizes amplitude to [-1.0, 1.0].

    Returns:
        audio: 1D numpy float32 array
        sample_rate: target sample rate (e.g. 16000)
        duration_seconds: audio duration in seconds
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(str(path), dtype="float32")

    # Convert to mono if multi-channel
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Simple resampling if sample rate doesn't match
    if sr != target_sample_rate:
        # Linear interpolation resampling (avoids heavy dependencies if librosa isn't present)
        duration = len(audio) / sr
        target_len = int(duration * target_sample_rate)
        indices = np.linspace(0, len(audio) - 1, target_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sr = target_sample_rate

    # Normalize amplitude
    max_amp = np.max(np.abs(audio))
    if max_amp > 1e-4:
        audio = audio / max_amp

    duration_sec = float(len(audio) / sr)
    return audio, sr, duration_sec


def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
