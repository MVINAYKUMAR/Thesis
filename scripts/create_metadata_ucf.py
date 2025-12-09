import os
import csv
import random
import argparse

def find_videos(root_dir, exts=(".mp4", ".avi", ".mkv", ".mov")):
    videos = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(exts):
                full_path = os.path.join(dirpath, f)
                videos.append(full_path)
    return videos

def infer_label_from_path(path):
    lower = path.lower()
    if os.sep + "nonfight" + os.sep in lower or os.sep + "normal" + os.sep in lower:
        return 0
    if os.sep + "fight" + os.sep in lower:
        return 1
    # Fallback: use folder name
    if "nonfight" in lower or "normal" in lower:
        return 0
    if "fight" in lower or "violence" in lower:
        return 1
    raise ValueError(f"Cannot infer label from: {path}")

def main():
    parser = argparse.ArgumentParser(description="Create metadata CSV for UCF violence subset.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder of UCF_subset (with Fight/NonFight folders).")
    parser.add_argument("--out_csv", type=str, default="data/metadata/ucf_subset_metadata.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    random.seed(args.seed)

    videos = find_videos(args.data_root)
    if not videos:
        print(f"No videos found under {args.data_root}")
        return

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    items = []
    for v in videos:
        label = infer_label_from_path(v)
        items.append((v, label))

    random.shuffle(items)

    n = len(items)
    n_train = int(args.train_ratio * n)
    n_val = int(args.val_ratio * n)
    n_test = n - n_train - n_val

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "filepath", "label", "split"])
        for idx, ((path, label), split) in enumerate(zip(items, splits)):
            vid = f"ucf_{idx:05d}"
            writer.writerow([vid, path, label, split])

    print(f"Wrote metadata for {n} videos to {args.out_csv}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

if __name__ == "__main__":
    main()
