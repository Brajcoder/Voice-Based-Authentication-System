"""
Audio preprocessing and cleaning module for Voice-Based Authentication.
Handles audio loading, mono conversion, resampling, silence removal,
amplitude normalization, and signal validation.
"""

import io
import os
import atexit
import tempfile
import threading
import numpy as np
import soundfile as sf
import librosa

# Track active temporary files for emergency cleanup on process exit
_ACTIVE_TEMP_FILES: set[str] = set()
_TEMP_FILES_LOCK = threading.Lock()


def _track_temp_file(path: str) -> None:
    """Register temporary file path for tracking."""
    with _TEMP_FILES_LOCK:
        _ACTIVE_TEMP_FILES.add(path)


def _cleanup_temp_file(path: str) -> None:
    """Remove a tracked temporary file and delete it from disk."""
    with _TEMP_FILES_LOCK:
        _ACTIVE_TEMP_FILES.discard(path)
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


@atexit.register
def cleanup_temp_files() -> None:
    """
    Module-level atexit handler that cleans up any remaining temporary files
    if the Flask process exits unexpectedly.
    """
    with _TEMP_FILES_LOCK:
        remaining_files = list(_ACTIVE_TEMP_FILES)
        _ACTIVE_TEMP_FILES.clear()

    for path in remaining_files:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass


def load_audio(audio_source, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio from various sources (bytes, file-like object, file path, or FileStorage).
    Converts to mono and resamples to target_sr.

    Args:
        audio_source: bytes, io.BytesIO, file path (str), or werkzeug FileStorage.
        target_sr: Target sample rate in Hz (default: 16000).

    Returns:
        tuple (signal, sr): 1D float32 numpy array and sample rate.
    """
    audio_bytes = None
    file_path = None

    if isinstance(audio_source, bytes):
        audio_bytes = audio_source
    elif hasattr(audio_source, "read"):
        # FileStorage or BytesIO
        audio_bytes = audio_source.read()
    elif isinstance(audio_source, str) and os.path.exists(audio_source):
        file_path = audio_source
    else:
        raise ValueError("Unsupported audio_source type or non-existent file path.")

    # Try loading from bytes directly with soundfile
    if audio_bytes is not None:
        try:
            bio = io.BytesIO(audio_bytes)
            data, sr = sf.read(bio, dtype="float32")
        except Exception:
            # Fallback: write to temp file and try librosa
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp_path = tmp.name
                    _track_temp_file(tmp_path)
                    tmp.write(audio_bytes)

                data, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
                return data.astype(np.float32), target_sr
            finally:
                if tmp_path:
                    _cleanup_temp_file(tmp_path)
    else:
        # Load directly from file path
        try:
            data, sr = sf.read(file_path, dtype="float32")
        except Exception:
            data, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return data.astype(np.float32), target_sr

    # Ensure 1D mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return data.astype(np.float32), sr


def remove_silence(signal: np.ndarray, top_db: float = 25.0) -> np.ndarray:
    """
    Trim leading and trailing silence from the audio signal using energy threshold.
    """
    if len(signal) == 0:
        return signal

    # librosa.effects.trim splits and trims silence below top_db
    try:
        trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
        if len(trimmed) > 0:
            return trimmed
    except Exception:
        pass
    return signal


def normalize_amplitude(signal: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize signal amplitude to prevent clipping and balance volume variations.
    """
    if len(signal) == 0:
        return signal

    max_val = np.max(np.abs(signal))
    if max_val > 1e-6:
        signal = signal * (target_peak / max_val)
    return signal


def clean_audio(
    signal: np.ndarray,
    sr: int = 16000,
    top_db: float = 25.0,
    min_duration: float = 0.5,
) -> np.ndarray:
    """
    Apply silence trimming, normalization, and quality validation.

    Args:
        signal: 1D numpy array of audio samples.
        sr: Sample rate (default: 16000).
        top_db: Decibels below peak to consider silence.
        min_duration: Minimum duration in seconds of clean audio required.

    Returns:
        Cleaned 1D numpy array.
    """
    # Remove silence
    trimmed = remove_silence(signal, top_db=top_db)

    # Check minimum duration
    duration = len(trimmed) / sr
    if duration < min_duration:
        raise ValueError(
            f"Voice recording too short ({duration:.2f}s). Please speak clearly for at least {min_duration} seconds."
        )

    # Normalize amplitude
    normalized = normalize_amplitude(trimmed)
    return normalized.astype(np.float32)


def preprocess_audio(audio_source, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    High-level entry point to load and clean an audio sample.
    """
    raw_signal, sr = load_audio(audio_source, target_sr=target_sr)
    cleaned_signal = clean_audio(raw_signal, sr=sr)
    return cleaned_signal, sr

