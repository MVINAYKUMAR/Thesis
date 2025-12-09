import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import classification_report, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fusion_dataset import RWFFusionDataset
from train_fusion import FusionModel


def get_loader(metadata_csv, pose_dir, split="test", clip_len=8, img_size=224,
               batch_size=8, num_workers=2):
    frame_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    ds = RWFFusionDataset(
        metadata_csv, pose_dir=pose_dir, split=split,
        clip_len=clip_len, img_size=img_size, transform=frame_transform
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return ds, loader


@torch.no_grad()
def eval_fusion(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    for clips, pose_seq, labels in loader:
        clips = clips.to(device)
        pose_seq = pose_seq.to(device)
        labels = labels.to(device)

        outputs = model(clips, pose_seq)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = (all_labels == all_preds).mean()
    return acc, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate fusion model on test set.")
    parser.add_argument("--metadata_csv", type=str, default="data/metadata/rwf2000_metadata.csv")
    parser.add_argument("--pose_dir", type=str, default="data/pose")
    parser.add_argument("--checkpoint", type=str, default="results/fusion_swin_pose/best_model.pth")
    parser.add_argument("--backbone", type=str, default="swin_tiny_patch4_window7_224")
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds, loader = get_loader(
        args.metadata_csv, args.pose_dir, split="test",
        clip_len=args.clip_len, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    print("Test samples:", len(ds))

    model = FusionModel(backbone_name=args.backbone, pose_dim=66,
                        pose_hidden=128, pose_out=128, num_classes=2).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} with val_acc={ckpt.get('val_acc', 'N/A')}")

    acc, labels, preds = eval_fusion(model, loader, device)
    print(f"\nTest Accuracy: {acc:.4f}")

    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix (rows = true, cols = pred):")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["NonFight", "Fight"]))


if __name__ == "__main__":
    main()
