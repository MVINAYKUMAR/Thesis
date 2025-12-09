import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2

import mediapipe as mp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_loader import RWFFightDataset  # to reuse sampling logic


def extract_pose_for_video(video_path, frame_indices, img_size=224):
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None  # will handle outside

    T = len(frame_indices)
    # 33 joints, 2 coords (x,y)
    pose_seq = np.zeros((T, 33 * 2), dtype=np.float32)

    for ti, fi in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        results = mp_pose.process(frame_rgb)
        if results.pose_landmarks:
            coords = []
            for lm in results.pose_landmarks.landmark:
                # normalize x,y by image width/height
                x = lm.x
                y = lm.y
                coords.extend([x, y])
            coords = np.array(coords, dtype=np.float32)
            if coords.shape[0] != 33 * 2:
                # something odd; skip
                continue
            pose_seq[ti] = coords

    cap.release()
    mp_pose.close()
    return pose_seq


def main():
    parser = argparse.ArgumentParser(description="Extract pose features for RWF-2000 videos.")
    parser.add_argument("--metadata_csv", type=str, default="data/metadata/rwf2000_metadata.csv")
    parser.add_argument("--out_dir", type=str, default="data/pose")
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--split", type=str, default="train,val,test")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    meta = pd.read_csv(args.metadata_csv)
    target_splits = [s.strip() for s in args.split.split(",")]

    # Create a temporary dataset just to reuse frame sampling method
    tmp_ds = RWFFightDataset(args.metadata_csv, split="train",
                             clip_len=args.clip_len, img_size=args.img_size)

    def sample_indices(num_frames):
        return tmp_ds._sample_frame_indices(num_frames)

    for _, row in meta.iterrows():
        if row["split"] not in target_splits:
            continue

        vid = row["video_id"]
        video_path = row["filepath"]
        out_path = os.path.join(args.out_dir, f"{vid}.npy")

        if os.path.exists(out_path):
            continue  # already done

        if not os.path.exists(video_path):
            print(f"[WARN] Video not found: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open video: {video_path}")
            cap.release()
            continue

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if num_frames <= 0:
            print(f"[WARN] No frames in video: {video_path}")
            continue

        frame_indices = sample_indices(num_frames)
        pose_seq = extract_pose_for_video(video_path, frame_indices, img_size=args.img_size)

        if pose_seq is None:
            print(f"[WARN] Pose extraction failed: {video_path}")
            continue

        # Save as (T, 66) float32
        np.save(out_path, pose_seq)
        print(f"Saved pose for {vid} -> {out_path}")

    print("Done pose extraction.")


if __name__ == "__main__":
    main()
