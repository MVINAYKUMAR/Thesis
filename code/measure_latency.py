import os
import sys
import time
import argparse
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from train_vit import ViTClipClassifier  # your existing class


def main():
    parser = argparse.ArgumentParser(description="Measure inference latency of ViTClipClassifier.")
    parser.add_argument("--backbone", type=str, required=True,
                        help="Backbone name, e.g. swin_tiny_patch4_window7_224 or maxvit_tiny_tf_224.in1k")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional: path to trained checkpoint (for realistic weights). If not provided, uses random weights.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to test on: 'cuda' or 'cpu'")
    parser.add_argument("--clip_len", type=int, default=8,
                        help="Number of frames per clip.")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size (H=W).")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warm-up iterations (not timed).")
    parser.add_argument("--iters", type=int, default=50,
                        help="Number of timed iterations.")
    args = parser.parse_args()

    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print("Using device:", device)

    # Build model
    model = ViTClipClassifier(backbone_name=args.backbone, num_classes=2).to(device)

    if args.checkpoint is not None:
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("No checkpoint provided, using randomly initialized weights.")

    model.eval()

    # Dummy clip: (B=1, T, C, H, W)
    dummy = torch.randn(1, args.clip_len, 3, args.img_size, args.img_size).to(device)

    # Warm-up
    print(f"Warm-up for {args.warmup} iterations...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy)

    # Timed runs
    print(f"Measuring latency over {args.iters} iterations...")
    times = []
    with torch.no_grad():
        for _ in range(args.iters):
            start = time.time()
            _ = model(dummy)
            end = time.time()
            times.append((end - start) * 1000.0)  # ms

    avg_ms = sum(times) / len(times)
    fps = 1000.0 / avg_ms  # clips per second (since 1 clip = 8 frames)

    print(f"\nAverage inference time per 8-frame clip: {avg_ms:.2f} ms")
    print(f"Approx. clip-level FPS: {fps:.2f} clips/sec")
    print(f"(Each clip has {args.clip_len} frames → effective {fps * args.clip_len:.2f} frames/sec)")


if __name__ == "__main__":
    main()
