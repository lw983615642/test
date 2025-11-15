import argparse
import os
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.models.polar_reflection_net import PolarReflectionRemovalNet


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("L")
    tensor = TF.to_tensor(img).unsqueeze(0)
    return tensor


def save_image(tensor: torch.Tensor, path: str) -> None:
    tensor = tensor.squeeze(0).clamp(0, 1)
    img = TF.to_pil_image(tensor)
    img.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the trained model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images captured through glass")
    parser.add_argument("--reflection_dir", type=str, required=True, help="Directory with reflection maps")
    parser.add_argument("--weights", type=str, required=True, help="Path to model checkpoint (best.pth)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save enhanced images")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)

    model = PolarReflectionRemovalNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    input_paths = sorted([p for p in Path(args.input_dir).iterdir() if p.is_file()])
    reflection_paths = sorted([p for p in Path(args.reflection_dir).iterdir() if p.is_file()])

    if len(input_paths) != len(reflection_paths):
        raise ValueError("Input and reflection directories must have the same number of images")

    os.makedirs(args.output_dir, exist_ok=True)

    for input_path, reflection_path in tqdm(zip(input_paths, reflection_paths), total=len(input_paths)):
        inp = load_image(str(input_path)).to(device)
        refl = load_image(str(reflection_path)).to(device)

        with torch.no_grad():
            clean, _, _ = model(inp, refl)
        save_image(clean.cpu(), str(Path(args.output_dir) / input_path.name))


if __name__ == "__main__":
    main()
