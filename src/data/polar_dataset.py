import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


def _list_sorted_files(directory: str, suffixes: Optional[Sequence[str]] = None) -> List[str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files = [f for f in files if os.path.isfile(f)]
    if suffixes:
        files = [f for f in files if os.path.splitext(f)[-1].lower() in suffixes]
    return sorted(files)


@dataclass
class PolarSample:
    input_path: str
    target_path: str
    reflection_path: str


class PolarReflectionDataset(Dataset):
    """Dataset for polarized reflection removal under low-light conditions.

    The dataset expects three directories containing aligned images:

    - inputs: captured through reflective glass (dark, reflective)
    - targets: clean reference capture without glass
    - reflections: isolated reflection image captured with black cloth

    Each directory must contain the same number of images with matching file names.
    The images are loaded as single-channel tensors in the range [0, 1].
    """

    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        reflection_dir: str,
        suffixes: Optional[Sequence[str]] = (".png", ".jpg", ".jpeg", ".tif", ".bmp"),
        augment: bool = True,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        super().__init__()
        self.input_paths = _list_sorted_files(input_dir, suffixes)
        self.target_paths = _list_sorted_files(target_dir, suffixes)
        self.reflection_paths = _list_sorted_files(reflection_dir, suffixes)

        if not (len(self.input_paths) == len(self.target_paths) == len(self.reflection_paths)):
            raise ValueError("Input, target, and reflection directories must contain the same number of images")

        self.augment = augment
        self.transform = transform

    def __len__(self) -> int:
        return len(self.input_paths)

    def _load_grayscale(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        tensor = TF.to_tensor(img)
        return tensor

    def _apply_random_augment(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            x = TF.hflip(x)
            y = TF.hflip(y)
            r = TF.hflip(r)
        if random.random() < 0.5:
            x = TF.vflip(x)
            y = TF.vflip(y)
            r = TF.vflip(r)
        if random.random() < 0.3:
            angle = random.choice([90, 180, 270])
            x = TF.rotate(x, angle)
            y = TF.rotate(y, angle)
            r = TF.rotate(r, angle)
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            x = x.clamp(1e-6, 1).pow(gamma)
        return x, y, r

    def __getitem__(self, idx: int) -> dict:
        input_tensor = self._load_grayscale(self.input_paths[idx])
        target_tensor = self._load_grayscale(self.target_paths[idx])
        reflection_tensor = self._load_grayscale(self.reflection_paths[idx])

        if self.augment:
            input_tensor, target_tensor, reflection_tensor = self._apply_random_augment(
                input_tensor, target_tensor, reflection_tensor
            )

        if self.transform:
            input_tensor, target_tensor, reflection_tensor = self.transform(
                input_tensor, target_tensor, reflection_tensor
            )

        return {
            "input": input_tensor,
            "target": target_tensor,
            "reflection": reflection_tensor,
            "input_path": self.input_paths[idx],
            "target_path": self.target_paths[idx],
            "reflection_path": self.reflection_paths[idx],
        }
