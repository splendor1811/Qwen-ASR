"""Audio and text utilities."""

from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def get_audio_duration(path: str) -> float:
    """Get duration of an audio file in seconds."""
    info = sf.info(path)
    return info.duration


def filter_by_duration(
    duration: float,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> bool:
    """Check if audio duration is within acceptable range."""
    return min_duration <= duration <= max_duration
