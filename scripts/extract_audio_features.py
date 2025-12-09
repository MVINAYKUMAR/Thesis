import os
import librosa
import numpy as np
import argparse

def extract_audio_features(path, sr=16000, n_mels=64):
    y, _ = librosa.load(path, sr=sr)
    if len(y) < sr:  # too short → pad
        y = np.pad(y, (0, sr-len(y)))
    
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # RMS Energy
    rms = librosa.feature.rms(y=y).mean()

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    return mel_db.astype(np.float32), np.array([rms, zcr], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--out_dir", default="data/audio")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.metadata_csv)

    os.makedirs(args.out_dir, exist_ok=True)

    for _, row in df.iterrows():
        video = row["filepath"]
        vid = row["video_id"]
        out_path = os.path.join(args.out_dir, f"{vid}.npz")

        try:
            mel, stats = extract_audio_features(video)
            np.savez(out_path, mel=mel, stats=stats)
            print(f"Saved audio for {vid}")
        except Exception as e:
            print(f"Failed {video}: {e}")

    print("Done audio extraction.")

if __name__ == "__main__":
    main()
