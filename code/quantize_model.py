import os
import sys
import argparse
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from train_vit import ViTClipClassifier  # uses your existing class


def main():
    parser = argparse.ArgumentParser(description="Dynamic quantization of ViTClipClassifier.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained best_model.pth checkpoint.")
    parser.add_argument("--backbone", type=str, required=True,
                        help="Backbone name, e.g. swin_tiny_patch4_window7_224 or maxvit_tiny_tf_224.in1k")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Output path for quantized model state dict, e.g. results/vit_swin_colab/quantized_model.pth")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Loading model on {device}...")

    # Load original model
    model = ViTClipClassifier(backbone_name=args.backbone, num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Apply dynamic quantization to Linear layers
    print("Applying dynamic quantization to Linear layers...")
    qmodel = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    # Save quantized state dict
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(qmodel.state_dict(), args.out_path)
    print(f"Quantized model state dict saved to: {args.out_path}")

    # Report size difference
    orig_size = os.path.getsize(args.checkpoint) / (1024 * 1024)
    quant_size = os.path.getsize(args.out_path) / (1024 * 1024)

    print(f"Original checkpoint size:  {orig_size:.2f} MB")
    print(f"Quantized state dict size: {quant_size:.2f} MB")
    print(f"Size reduction: {orig_size - quant_size:.2f} MB (~{(1 - quant_size/orig_size) * 100:.1f}% smaller)")


if __name__ == "__main__":
    main()
