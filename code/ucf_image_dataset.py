import os
import sys
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms as T

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

class UCFImageDataset(Dataset):
    """
    Simple image dataset based on PNG frames.
    Expects metadata CSV with:
      image_id, filepath, label, split

    Returns:
      img_tensor: (C, H, W)
      label: Long tensor
    """

    def __init__(self, metadata_csv, split="train", img_size=224, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.img_size = img_size

        if transform is None:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        path = row["filepath"]
        label = int(row["label"])

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # fallback: blank image
            img = Image.new("RGB", (self.img_size, self.img_size))

        img_tensor = self.transform(img)
        return img_tensor, torch.tensor(label, dtype=torch.long)
