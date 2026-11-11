"""
CNN-based speaker embedding model and verification utilities.
Provides:
- PyTorch VoiceEmbeddingCNN architecture
- High-level extract_embedding() function
- Cosine similarity computation
- Embedding serialization for SQLite storage
"""

import io
import soundfile as sf
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.features import extract_mfcc, extract_acoustic_stats


class VoiceEmbeddingCNN(nn.Module):
    """
    Convolutional Neural Network for Speaker Voice Embedding extraction.
    Takes multi-channel audio features (MFCC, delta, delta2) and produces
    a normalized, compact 128-dimensional speaker embedding vector.
    Preserves frequency characteristics and uses temporal statistical pooling
    (mean and standard deviation) for duration-invariance.
    """

    def __init__(self, in_channels: int = 3, embedding_dim: int = 128):
        super(VoiceEmbeddingCNN, self).__init__()

        # Conv Block 1: frequency 40 -> 20
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv Block 2: frequency 20 -> 10
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv Block 3: frequency 10 -> 5
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Temporal statistical pooling aggregates mean + std over time frames (dim=3),
        # leaving frequency (5 bins) and channels (64): 64 * 5 * 2 = 640
        self.fc1 = nn.Linear(64 * 5 * 2, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, 3, n_mfcc=40, time_frames=T)
        """
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))  # freq 40 -> 20

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))  # freq 20 -> 10

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))  # freq 10 -> 5

        # Temporal Statistical Pooling (mean and standard deviation along time dim=3)
        mean = torch.mean(x, dim=3)  # shape: (batch, 64, 5)
        std = torch.std(x, dim=3, unbiased=False) + 1e-6  # shape: (batch, 64, 5)

        # Concatenate along channel dim: shape (batch, 128, 5) -> flatten: (batch, 640)
        stats = torch.cat([mean, std], dim=1).flatten(1)

        # Projection to embedding space
        h = F.relu(self.bn_fc(self.fc1(stats)))
        h = self.dropout(h)
        emb = self.fc2(h)

        # L2 normalize the embedding
        emb = F.normalize(emb, p=2, dim=1)
        return emb


import os

# Global singleton instance of the model
_MODEL_INSTANCE: VoiceEmbeddingCNN | None = None


def get_model() -> VoiceEmbeddingCNN:
    """Return or initialize the singleton VoiceEmbeddingCNN model."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        torch.manual_seed(42)  # Deterministic initialization
        model = VoiceEmbeddingCNN(in_channels=3, embedding_dim=128)

        weights_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict)
                print(f"[ML] Loaded trained weights from {weights_path}")
            except Exception as e:
                print(f"[ML] Could not load weights from {weights_path}: {e}")

        model.eval()
        _MODEL_INSTANCE = model
    return _MODEL_INSTANCE


from resemblyzer import VoiceEncoder, preprocess_wav

_VOICE_ENCODER_INSTANCE: VoiceEncoder | None = None


def get_voice_encoder() -> VoiceEncoder:
    """Return or initialize the singleton Resemblyzer VoiceEncoder model."""
    global _VOICE_ENCODER_INSTANCE
    if _VOICE_ENCODER_INSTANCE is None:
        _VOICE_ENCODER_INSTANCE = VoiceEncoder()
    return _VOICE_ENCODER_INSTANCE


def extract_embedding_cnn(audio_input, sr: int = 16000) -> np.ndarray:
    """
    Extract a normalized 128-dimensional speaker embedding using the trained VoiceEmbeddingCNN.
    Accepts file path (str), file-like object, or 1D float32 numpy array.
    """
    if isinstance(audio_input, str):
        data, orig_sr = sf.read(audio_input, stop=40000, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if orig_sr != sr:
            data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
        signal = data
    else:
        signal = audio_input

    if len(signal) < 1024:
        signal = np.pad(signal, (0, 1024 - len(signal)), mode="constant")

    feat = extract_mfcc(signal, sr=sr, n_mfcc=40, include_deltas=True)
    mfcc = feat[0:40, :]
    delta = feat[40:80, :]
    delta2 = feat[80:120, :]
    tensor_input = np.stack([mfcc, delta, delta2], axis=0)  # (3, 40, T)
    x = torch.from_numpy(tensor_input).unsqueeze(0).float()

    model = get_model()
    with torch.no_grad():
        cnn_out = model(x)

    cnn_emb = cnn_out.squeeze(0).cpu().numpy().astype(np.float32)
    stats = extract_acoustic_stats(signal, sr=sr)
    stats_norm = (stats - np.mean(stats)) / (np.std(stats) + 1e-8)

    combined = np.concatenate([cnn_emb[:120], stats_norm[:8]], axis=0).astype(np.float32)
    norm = np.linalg.norm(combined)
    if norm > 1e-6:
        combined = combined / norm
    return combined


def extract_embedding(audio_input, sr: int = 16000) -> np.ndarray:
    """
    Extract a normalized 256-dimensional speaker embedding using Resemblyzer VoiceEncoder.
    Accepts file path (str), file-like object, or 1D float32 numpy array.
    """
    encoder = get_voice_encoder()
    try:
        if isinstance(audio_input, str):
            wav = preprocess_wav(audio_input)
        else:
            wav = preprocess_wav(audio_input, source_sr=sr)
    except Exception:
        wav = audio_input

    if len(wav) == 0:
        wav = audio_input

    embedding = encoder.embed_utterance(wav)
    embedding = np.asarray(embedding, dtype=np.float32)

    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm
    return embedding


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two voice embeddings.
    Cosine similarity of normalized vectors is their dot product.
    Returns float in range [-1.0, 1.0].
    """
    e1 = np.asarray(embedding1, dtype=np.float32).flatten()
    e2 = np.asarray(embedding2, dtype=np.float32).flatten()

    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    dot_product = float(np.dot(e1, e2) / (norm1 * norm2))
    # Clamp to [-1.0, 1.0]
    return max(-1.0, min(1.0, dot_product))


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialize numpy embedding array to binary bytes for SQLite BLOB storage."""
    arr = np.asarray(embedding, dtype=np.float32)
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return bio.getvalue()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    """Deserialize binary bytes from SQLite BLOB to numpy embedding array."""
    bio = io.BytesIO(data)
    arr = np.load(bio, allow_pickle=False)
    return arr.astype(np.float32)
