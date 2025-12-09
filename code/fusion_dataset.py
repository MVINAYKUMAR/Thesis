import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_loader import RWFFightDataset


class RWFFusionDataset(Dataset):
    """
    Returns (clip, pose_seq, label)
    clip: (T, C, H, W)
    pose_seq: (T, 66)
    """

    def __init__(self, metadata_csv, pose_dir="data/pose", split="train",
                 clip_len=8, img_size=224, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.pose_dir = pose_dir
        self.clip_len = clip_len
        self.img_size = img_size
        self.transform = transform

        # We reuse RWFFightDataset just for visual clip loading
        self.visual_ds = RWFFightDataset(metadata_csv, split=split,
                                         clip_len=clip_len, img_size=img_size,
                                         transform=transform)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        vid = row["video_id"]
        label = int(row["label"])

        # Visual clip
        clip_tensor, _ = self.visual_ds[idx]  # (T, C, H, W)

        # Pose
        pose_path = os.path.join(self.pose_dir, f"{vid}.npy")
        if os.path.exists(pose_path):
            pose_seq = np.load(pose_path).astype(np.float32)  # (T, 66)
        else:
            # Fallback: zeros
            pose_seq = np.zeros((self.clip_len, 66), dtype=np.float32)

        # Ensure correct shape
        if pose_seq.shape[0] != self.clip_len:
            # pad or crop
            Tcur = pose_seq.shape[0]
            if Tcur > self.clip_len:
                pose_seq = pose_seq[:self.clip_len]
            else:
                pad = np.tile(pose_seq[-1:], (self.clip_len - Tcur, 1))
                pose_seq = np.concatenate([pose_seq, pad], axis=0)

        pose_tensor = torch.from_numpy(pose_seq)  # (T, 66)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return clip_tensor, pose_tensor, label_tensor
