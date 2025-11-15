import os
from contextlib import contextmanager
from typing import Dict, Iterable

import torch


def seed_everything(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class MetricTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.values: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, key: str, value: float, n: int = 1) -> None:
        if key not in self.values:
            self.values[key] = 0.0
            self.counts[key] = 0
        self.values[key] += value * n
        self.counts[key] += n

    def average(self, key: str) -> float:
        return self.values.get(key, 0.0) / max(self.counts.get(key, 1), 1)

    def averages(self) -> Dict[str, float]:
        return {k: self.average(k) for k in self.values}


@contextmanager
def torch_autocast(device_type: str = "cuda", enabled: bool = True):
    if enabled:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            yield
    else:
        yield


def save_checkpoint(state: Dict, is_best: bool, checkpoint_dir: str, filename: str = "checkpoint.pth") -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best.pth"))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
