"""
Dataset module for Voice-Based Authentication System.
Handles audio feature loading, caching, and data augmentation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class VoxCelebDataset(Dataset):
    """
    Dataset for speaker identification audio features with data augmentation.
    """

    def __init__(self, features, labels, augment: bool = False):
        """
        Args:
            features: numpy array of shape (N, channels, freq, time)
            labels: numpy array or list of integer speaker IDs
            augment: If True, applies random noise augmentation during training
        """
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64)).long()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx].copy()

        # Add noise augmentation to __getitem__
        if self.augment:
            noise = np.random.randn(*features.shape).astype(np.float32) * 0.005
            features = features + noise

        return torch.from_numpy(features).float(), self.labels[idx]

