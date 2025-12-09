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
    # Adjust to match RWF-2000: NonFight vs Fight
    if "nonfight" in lower or "non-fight" in lower:
        return 0
    elif "fight" in lower or "violence" in lower:
        return 1
    else:
        raise ValueError(f"Cannot infer label from path: {path}")

def main():
    parser = argparse.ArgumentParser(description="Create 70/15/15 metadata CSV for RWF-2000.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder containing RWF-2000 (train/val/Fight/NonFight).")
    parser.add_argument("--out_csv", type=str, default="data/metadata/rwf2000_metadata.csv",
                        help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    videos = find_videos(args.data_root)
    if not videos:
        print(f"No videos found under {args.data_root}")
        return

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Build list of (path, label)
    items = []
    for v in videos:
        label = infer_label_from_path(v)
        items.append((v, label))

    # Shuffle
    random.shuffle(items)

    n = len(items)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    # Use UTF-8 to avoid Windows cp1252 issues
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "filepath", "label", "split"])
        for idx, ((path, label), split) in enumerate(zip(items, splits)):
            vid = f"vid_{idx:05d}"
            writer.writerow([vid, path, label, split])

    print(f"Wrote metadata for {n} videos to {args.out_csv}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

if __name__ == "__main__":
    main()
