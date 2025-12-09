import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Ensure we can import data_loader and model class from this folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_loader import RWFFightDataset
from train_vit import ViTClipClassifier  # reuse same class


def get_dataloader(metadata_csv, split="test", clip_len=8, img_size=224, batch_size=4, num_workers=4):
    frame_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    ds = RWFFightDataset(
        metadata_csv,
        split=split,
        clip_len=clip_len,
        img_size=img_size,
        transform=frame_transform
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return ds, loader


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        outputs = model(clips)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = (all_labels == all_preds).mean()
    return acc, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate best ViT model on test set.")
    parser.add_argument("--metadata_csv", type=str, default="data/metadata/rwf2000_metadata.csv")
    parser.add_argument("--checkpoint", type=str, default="results/vit_baseline/best_model.pth")
    parser.add_argument("--backbone", type=str, default="swin_tiny_patch4_window7_224")
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data
    test_ds, test_loader = get_dataloader(
        args.metadata_csv,
        split="test",
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print("Test samples:", len(test_ds))

    # Model
    model = ViTClipClassifier(backbone_name=args.backbone, num_classes=2).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} with val_acc={ckpt.get('val_acc', 'N/A')}")

    # Evaluate
    acc, labels, preds = evaluate_test(model, test_loader, device)
    print(f"\nTest Accuracy: {acc:.4f}")

    # Confusion matrix and report
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix (rows = true, cols = pred):")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["NonFight", "Fight"]))


if __name__ == "__main__":
    main()
