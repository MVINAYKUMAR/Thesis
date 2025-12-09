import os
import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

import timm
import numpy as np

# Make sure Python can import modules from this folder (code/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_loader import RWFFightDataset


class ViTClipClassifier(nn.Module):
    """
    Wraps a pretrained ViT/Swin from timm and applies it frame-wise to a clip,
    then averages features over time and classifies Fight / Non-Fight.
    Input: (B, T, C, H, W)
    """
    def __init__(self, backbone_name="swin_tiny_patch4_window7_224", num_classes=2):
        super().__init__()
        # num_classes=0 gives a feature extractor (no classifier head)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        self.feat_dim = self.backbone.num_features
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)         # (B*T, C, H, W)
        feats = self.backbone(x)           # (B*T, F)
        feats = feats.view(B, T, -1)       # (B, T, F)
        clip_feat = feats.mean(dim=1)      # (B, F)
        logits = self.classifier(clip_feat)  # (B, num_classes)
        return logits


def get_dataloaders(metadata_csv, clip_len=8, img_size=224,
                    batch_size=4, num_workers=4):

    # Standard ImageNet mean/std for timm models
    frame_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    train_ds = RWFFightDataset(
        metadata_csv,
        split="train",
        clip_len=clip_len,
        img_size=img_size,
        transform=frame_transform
    )
    val_ds = RWFFightDataset(
        metadata_csv,
        split="val",
        clip_len=clip_len,
        img_size=img_size,
        transform=frame_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in loader:
        clips = clips.to(device)  # (B, T, C, H, W)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        outputs = model(clips)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * clips.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train ViT on RWF-2000 clips.")
    parser.add_argument("--metadata_csv", type=str,
                        default="data/metadata/rwf2000_metadata.csv",
                        help="Path to metadata CSV.")
    parser.add_argument("--backbone", type=str,
                        default="swin_tiny_patch4_window7_224",
                        help="timm backbone name.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="results/vit_baseline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Seed (optional but recommended)
    torch.manual_seed(42)
    np.random.seed(42)

    # Dataloaders
    train_loader, val_loader = get_dataloaders(
        args.metadata_csv,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model, loss, optimizer
    model = ViTClipClassifier(backbone_name=args.backbone, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    history_path = os.path.join(args.out_dir, "training_log.csv")

    # Write CSV header
    with open(history_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"  Train loss: {train_loss:.4f}  |  Train acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  |  Val   acc: {val_acc:.4f}")

        # Append metrics
        with open(history_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.out_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  ✅ New best model saved to {ckpt_path} (val_acc={val_acc:.4f})")

    print("\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
