"""
Training script for Voice-Based Authentication System.
Trains the VoiceEmbeddingCNN backbone on speaker identification using VoxCeleb dataset.
"""

import os
import sys

# Ensure immediate unbuffered output in terminal and logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import glob
import time
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml.model import VoiceEmbeddingCNN
from ml.dataset import VoxCelebDataset

# ----------------- Configuration -----------------
DATASET_PATH = r"D:\vox1_test_wav\wav"
if not os.path.exists(DATASET_PATH):
    # Fallback if unzipped directly without wav subfolder
    alt_path = r"D:\vox1_test_wav"
    if os.path.exists(alt_path):
        DATASET_PATH = alt_path

max_speakers = 35
samples_per_speaker = 80

EPOCHS = 30
LR = 0.0005
BATCH_SIZE = 16
SAMPLE_RATE = 16000
TARGET_FRAMES = 80
# -------------------------------------------------


def load_feature_from_wav(file_path: str, sr: int = 16000, target_frames: int = 80) -> np.ndarray:
    """
    Read audio and extract (3, 40, target_frames) MFCC + delta + delta2 tensor.
    Reads first 2.5 seconds (40,000 samples) for fast, standardized representation.
    """
    try:
        data, orig_sr = sf.read(file_path, stop=40000, dtype="float32")
    except Exception:
        data, orig_sr = librosa.load(file_path, sr=sr, duration=2.5, mono=True)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if orig_sr != sr:
        data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)

    # Silence/short audio padding
    if len(data) < 1024:
        data = np.pad(data, (0, 1024 - len(data)), mode="constant")

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40, n_fft=1024, hop_length=512)
    # CMVN (mean-variance normalization)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    feat = np.stack([mfcc, d1, d2], axis=0)  # Shape: (3, 40, T)

    # Pad or slice to target_frames
    cur_frames = feat.shape[2]
    if cur_frames < target_frames:
        feat = np.pad(feat, ((0, 0), (0, 0), (0, target_frames - cur_frames)), mode="constant")
    else:
        feat = feat[:, :, :target_frames]

    return feat.astype(np.float32)


class SpeakerClassifier(nn.Module):
    """Classification model wrapping VoiceEmbeddingCNN backbone with dropout regularization."""

    def __init__(self, backbone: VoiceEmbeddingCNN, num_classes: int):
        super(SpeakerClassifier, self).__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        emb = self.backbone(x)
        logits = self.classifier(self.drop(emb))
        return logits, emb


def prepare_dataset(dataset_path: str, max_spk: int, max_samples: int):
    print("=" * 65)
    print(f" Scanning dataset from: {dataset_path}")
    print(f" Target configuration: max_speakers={max_spk}, samples_per_speaker={max_samples}")
    print("=" * 65)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Discover speaker directories
    all_subdirs = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    all_subdirs.sort()
    selected_speakers = all_subdirs[:max_spk]
    num_classes = len(selected_speakers)

    print(f" Found {len(all_subdirs)} total speakers. Selected first {num_classes} speakers.")

    file_list = []
    labels_list = []

    for spk_idx, spk_id in enumerate(selected_speakers):
        spk_dir = os.path.join(dataset_path, spk_id)
        wav_files = glob.glob(os.path.join(spk_dir, "**", "*.wav"), recursive=True)
        wav_files.sort()
        chosen = wav_files[:max_samples]
        for f in chosen:
            file_list.append(f)
            labels_list.append(spk_idx)

    total_samples = len(file_list)
    print(f" Total audio files to load: {total_samples}")
    print(" Pre-extracting MFCC acoustic features into memory...")

    t0 = time.time()
    feature_list = []
    for idx, fpath in enumerate(file_list):
        feat = load_feature_from_wav(fpath, sr=SAMPLE_RATE, target_frames=TARGET_FRAMES)
        feature_list.append(feat)
        if (idx + 1) % 500 == 0 or (idx + 1) == total_samples:
            print(f"  Processed {idx + 1}/{total_samples} audio samples ({round(time.time() - t0, 1)}s)")

    features_array = np.stack(feature_list, axis=0)  # Shape: (N, 3, 40, target_frames)
    labels_array = np.array(labels_list, dtype=np.int64)

    # Shuffle and train/val split (85% train, 15% val)
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.85 * total_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    # Train dataset with noise data augmentation; Val dataset without augmentation
    train_ds = VoxCelebDataset(features_array[train_idx], labels_array[train_idx], augment=True)
    val_ds = VoxCelebDataset(features_array[val_idx], labels_array[val_idx], augment=False)

    print(f" Feature extraction complete: {len(train_ds)} train samples (augmented), {len(val_ds)} validation samples.\n")
    return train_ds, val_ds, num_classes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, num_classes = prepare_dataset(DATASET_PATH, max_speakers, samples_per_speaker)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize backbone and classifier
    torch.manual_seed(42)
    backbone = VoiceEmbeddingCNN(in_channels=3, embedding_dim=128)
    model = SpeakerClassifier(backbone, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_acc = 0.0
    output_weights_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")

    print("=" * 65)
    print(" Starting CNN Speaker Identification Training")
    print("=" * 65)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits, _ = model(x_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100.0
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch:2d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save only the backbone feature extractor weights
            torch.save(backbone.state_dict(), output_weights_path)

    print("\n" + "=" * 65)
    print(f" Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f" Saved backbone weights to: {output_weights_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
