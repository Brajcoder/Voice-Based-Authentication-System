"""
Audio feature extraction module for Voice-Based Authentication.
Extracts MFCCs, delta, delta-delta features, and Mel-spectrograms.
"""

import numpy as np
import librosa


def extract_mfcc(
    signal: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 512,
    include_deltas: bool = True,
) -> np.ndarray:
    """
    Extract MFCC features and optionally delta and delta-delta coefficients.

    Args:
        signal: 1D float32 audio signal.
        sr: Sample rate (default 16000).
        n_mfcc: Number of MFCC coefficients to compute (default 40).
        n_fft: FFT window size.
        hop_length: Number of audio samples between adjacent frames.
        include_deltas: If True, appends 1st and 2nd temporal derivatives (deltas).

    Returns:
        2D numpy array of shape (feature_dim, time_frames).
        If include_deltas is True, feature_dim = n_mfcc * 3 (e.g. 120).
    """
    if len(signal) < n_fft:
        # Pad short signals
        signal = np.pad(signal, (0, n_fft - len(signal)), mode="constant")

    # Compute base MFCCs
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Cepstral Mean and Variance Normalization (CMVN)
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (
        np.std(mfccs, axis=1, keepdims=True) + 1e-8
    )

    if not include_deltas:
        return mfccs.astype(np.float32)

    # Compute deltas
    delta = librosa.feature.delta(mfccs, order=1)
    delta2 = librosa.feature.delta(mfccs, order=2)

    # Stack vertically: shape (n_mfcc * 3, time_frames)
    combined = np.vstack([mfccs, delta, delta2])
    return combined.astype(np.float32)


def extract_mel_spectrogram(
    signal: np.ndarray,
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Extract log Mel-spectrogram features.

    Returns:
        2D numpy array of shape (n_mels, time_frames).
    """
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)), mode="constant")

    melspec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    # Convert power spectrogram to decibel units
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    return log_melspec.astype(np.float32)


def extract_acoustic_stats(signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract high-level acoustic descriptors (spectral centroid, bandwidth,
    rolloff, and zero-crossing rate) that characterize individual speaker vocal tract.
    """
    if len(signal) < 1024:
        signal = np.pad(signal, (0, 1024 - len(signal)), mode="constant")

    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=signal)

    stats = np.array([
        np.mean(centroid),
        np.std(centroid),
        np.mean(bandwidth),
        np.std(bandwidth),
        np.mean(rolloff),
        np.std(rolloff),
        np.mean(zcr),
        np.std(zcr),
    ], dtype=np.float32)

    return stats


def extract_fixed_feature_tensor(
    signal: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    target_frames: int = 100,
) -> np.ndarray:
    """
    Extract MFCC features and adjust time dimension to a fixed size (target_frames)
    via padding or truncation. Useful for feeding fixed-size batches to CNNs.

    Returns:
        3D numpy array of shape (3, n_mfcc, target_frames), where 3 corresponds to
        [mfcc, delta, delta2].
    """
    if len(signal) < 1024:
        signal = np.pad(signal, (0, 1024 - len(signal)), mode="constant")

    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=512)
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
    delta = librosa.feature.delta(mfccs, order=1)
    delta2 = librosa.feature.delta(mfccs, order=2)

    # Shape: (3, n_mfcc, time_frames)
    feat = np.stack([mfccs, delta, delta2], axis=0)

    _, _, cur_frames = feat.shape
    if cur_frames < target_frames:
        pad_width = target_frames - cur_frames
        feat = np.pad(feat, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
    elif cur_frames > target_frames:
        # Take central frames
        start = (cur_frames - target_frames) // 2
        feat = feat[:, :, start : start + target_frames]

    return feat.astype(np.float32)
