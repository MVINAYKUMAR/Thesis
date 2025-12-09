import os
import csv
import random
import argparse

def find_images(root_dir, exts=(".png", ".jpg", ".jpeg")):
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(exts):
                full_path = os.path.join(dirpath, f)
                images.append(full_path)
    return images

def infer_label_from_path(path):
    lower = path.lower()
    if os.sep + "nonfight" + os.sep in lower or os.sep + "normal" + os.sep in lower:
        return 0
    if os.sep + "fight" + os.sep in lower or os.sep + "violence" + os.sep in lower:
        return 1
    # Fallback based on folder name
    if "nonfight" in lower or "normal" in lower:
        return 0
    if "fight" in lower or "violence" in lower:
        return 1
    raise ValueError(f"Cannot infer label from: {path}")

def main():
    parser = argparse.ArgumentParser(description="Create metadata CSV for UCF PNG frame dataset.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder of UCF_png (with Fight/NonFight).")
    parser.add_argument("--out_csv", type=str, default="data/metadata/ucf_images_metadata.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    random.seed(args.seed)

    images = find_images(args.data_root)
    if not images:
        print(f"No images found under {args.data_root}")
        return

    items = []
    for img_path in images:
        label = infer_label_from_path(img_path)
        items.append((img_path, label))

    random.shuffle(items)

    n = len(items)
    n_train = int(args.train_ratio * n)
    n_val = int(args.val_ratio * n)
    n_test = n - n_train - n_val

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "filepath", "label", "split"])
        for idx, ((path, label), split) in enumerate(zip(items, splits)):
            img_id = f"uimg_{idx:06d}"
            writer.writerow([img_id, path, label, split])

    print(f"Wrote metadata for {n} images to {args.out_csv}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

if __name__ == "__main__":
    main()
