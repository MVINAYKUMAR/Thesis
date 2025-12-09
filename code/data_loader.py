
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class RWFFightDataset(Dataset):
    """RWF-2000 dataset loader that reads short clips on the fly using OpenCV.

    Expects a metadata CSV with columns: video_id, filepath, label, split
    """

    def __init__(self, metadata_csv, split="train", clip_len=8, img_size=224, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.clip_len = clip_len
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def _sample_frame_indices(self, num_frames):
        """Sample clip_len indices uniformly from the video frame range.
        If video has fewer frames than clip_len, repeat frames.
        """
        if num_frames <= 0:
            # Fallback: dummy single frame
            return np.zeros(self.clip_len, dtype=int)

        if num_frames >= self.clip_len:
            # linspace over [0, num_frames-1]
            indices = np.linspace(0, num_frames - 1, num=self.clip_len)
            indices = np.round(indices).astype(int)
        else:
            # Not enough frames: repeat
            base = np.arange(num_frames)
            reps = int(np.ceil(self.clip_len / num_frames))
            tiled = np.tile(base, reps)
            indices = tiled[:self.clip_len]
        return indices

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        video_path = row["filepath"]
        label = int(row["label"])

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._sample_frame_indices(num_frames)

        frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret:
                # If we fail to read, reuse last valid frame or create black
                if len(frames) > 0:
                    frame = frames[-1]
                else:
                    frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)

        cap.release()

        clip = np.stack(frames, axis=0)  # (T, H, W, C)

        if self.transform is not None:
            # If using torchvision transforms expecting PIL, apply per frame
            processed = []
            for f in clip:
                processed.append(self.transform(f))
            clip_tensor = torch.stack(processed, dim=0)  # (T, C, H, W)
        else:
            # Convert to tensor and normalize to [0,1]
            clip = clip.astype(np.float32) / 255.0
            clip = np.transpose(clip, (0, 3, 1, 2))  # (T, C, H, W)
            clip_tensor = torch.from_numpy(clip)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return clip_tensor, label_tensor

if __name__ == "__main__":
    # Simple smoke test (adjust paths before running):
    csv_path = "data/metadata/rwf2000_metadata.csv"
    if os.path.exists(csv_path):
        ds = RWFFightDataset(csv_path, split="train", clip_len=8, img_size=224)
        print("Dataset size:", len(ds))
        clip, y = ds[0]
        print("Clip shape:", clip.shape)  # Expected: (T, C, H, W)
        print("Label:", y)
    else:
        print(f"Metadata CSV not found at {csv_path}. Run create_metadata.py first.")
