import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import timm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fusion_dataset import RWFFusionDataset
from pose_encoder import PoseEncoder


class VisualBackbone(nn.Module):
    """
    Frame-wise ViT backbone -> avg over time.
    Input: (B, T, C, H, W)
    Output: (B, F)
    """
    def __init__(self, backbone_name="swin_tiny_patch4_window7_224"):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        self.feat_dim = self.backbone.num_features

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)           # (B*T, F)
        feats = feats.view(B, T, -1)       # (B, T, F)
        clip_feat = feats.mean(dim=1)      # (B, F)
        return clip_feat


class FusionModel(nn.Module):
    """
    Late fusion: concat visual feat + pose feat -> classifier
    """
    def __init__(self, backbone_name="swin_tiny_patch4_window7_224",
                 pose_dim=66, pose_hidden=128, pose_out=128, num_classes=2):
        super().__init__()
        self.visual = VisualBackbone(backbone_name=backbone_name)
        self.pose_enc = PoseEncoder(input_dim=pose_dim, hidden_dim=pose_hidden, out_dim=pose_out)
        fusion_dim = self.visual.feat_dim + pose_out
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, clips, pose_seq):
        # clips: (B, T, C, H, W)
        # pose_seq: (B, T, 66)
        visual_feat = self.visual(clips)       # (B, Fv)
        pose_feat = self.pose_enc(pose_seq)    # (B, Fp)
        fused = torch.cat([visual_feat, pose_feat], dim=1)
        logits = self.classifier(fused)
        return logits


def get_dataloaders(metadata_csv, pose_dir, clip_len=8, img_size=224,
                    batch_size=4, num_workers=4):

    frame_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    train_ds = RWFFusionDataset(
        metadata_csv, pose_dir=pose_dir, split="train",
        clip_len=clip_len, img_size=img_size, transform=frame_transform
    )
    val_ds = RWFFusionDataset(
        metadata_csv, pose_dir=pose_dir, split="val",
        clip_len=clip_len, img_size=img_size, transform=frame_transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for clips, pose_seq, labels in loader:
        clips = clips.to(device)              # (B, T, C, H, W)
        pose_seq = pose_seq.to(device)        # (B, T, 66)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips, pose_seq)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for clips, pose_seq, labels in loader:
        clips = clips.to(device)
        pose_seq = pose_seq.to(device)
        labels = labels.to(device)

        outputs = model(clips, pose_seq)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * clips.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train fusion (visual + pose) model on RWF-2000.")
    parser.add_argument("--metadata_csv", type=str, default="data/metadata/rwf2000_metadata.csv")
    parser.add_argument("--pose_dir", type=str, default="data/pose")
    parser.add_argument("--backbone", type=str, default="swin_tiny_patch4_window7_224")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="results/fusion_swin_pose")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, val_loader = get_dataloaders(
        args.metadata_csv, args.pose_dir,
        clip_len=args.clip_len, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = FusionModel(
        backbone_name=args.backbone,
        pose_dim=66,
        pose_hidden=128,
        pose_out=128,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    log_path = os.path.join(args.out_dir, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"  Train loss: {train_loss:.4f}  |  Train acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  |  Val   acc: {val_acc:.4f}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

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
