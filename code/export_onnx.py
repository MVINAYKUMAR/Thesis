import torch
import argparse
from train_vit import ViTClipClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="maxvit_tiny_tf_224.in1k")
    parser.add_argument("--out", default="vit_maxvit_rwf.onnx")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = ViTClipClassifier(backbone_name=args.backbone, num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 8, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["clip"],
        output_names=["logits"],
        opset_version=17,           # 👈 bumped from 13 → 17
    )

    print(f"Exported ONNX model to {args.out}")

if __name__ == "__main__":
    main()
