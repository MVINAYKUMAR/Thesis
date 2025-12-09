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


# def infer_label_from_name(filename):
#     """
#     Returns 1 for fight, 0 for non-fight.
#     Adjust rules depending on your actual naming.
#     """
#     name = os.path.basename(filename).lower()

#     # Common patterns: "fight", "fi", "Fight", etc.
#     # And "nonfight", "no_fight", "nofight", "no" for negatives
#     # You can print some names and adjust if needed.
#     if "nonfight" in name or "no_fight" in name or "no-fight" in name or "nofight" in name:
#         return 0
#     if name.startswith("no") and "fight" not in name:
#         # e.g., no001.avi
#         return 0

#     if "fight" in name or name.startswith("fi"):
#         return 1

#     # If we cannot infer, raise so we notice
#     raise ValueError(f"Cannot infer label from filename: {filename}")

def infer_label_from_name(filepath):
    """
    Returns 1 for fight, 0 for non-fight.
    Uses folder names first (Fight/NonFight), then falls back to filename.
    """
    lower_path = filepath.lower()
    base = os.path.basename(filepath).lower()

    # 1) Use folder names first
    if os.sep + "nonfight" + os.sep in lower_path or os.sep + "non_fight" + os.sep in lower_path:
        return 0
    if os.sep + "fight" + os.sep in lower_path:
        return 1

    # 2) Also treat 'normal' or 'burglary' etc under NonFight if you want
    if os.sep + "normal" + os.sep in lower_path or os.sep + "burglary" + os.sep in lower_path:
        return 0

    # 3) Fallback: old filename-based rules
    if "nonfight" in base or "no_fight" in base or "no-fight" in base or "nofight" in base:
        return 0
    if base.startswith("no") and "fight" not in base:
        return 0
    if "fight" in base or base.startswith("fi"):
        return 1

    # 4) If still ambiguous, fail so we can see what's wrong
    raise ValueError(f"Cannot infer label from filename/path: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Create metadata CSV for Hockey Fight dataset.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder containing all Hockey Fight videos.")
    parser.add_argument("--out_csv", type=str, default="data/metadata/hockey_metadata.csv",
                        help="Output CSV path.")
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
        label = infer_label_from_name(v)
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
            vid = f"hf_{idx:05d}"
            writer.writerow([vid, path, label, split])

    print(f"Wrote metadata for {n} videos to {args.out_csv}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")


if __name__ == "__main__":
    main()
