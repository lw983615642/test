from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """Structural similarity index as a loss (1 - SSIM)."""

    def __init__(self, window_size: int = 11, channel: int = 1, reduction: str = "mean") -> None:
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.reduction = reduction
        self.register_buffer("window", self._create_window(window_size, channel))

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        gauss = torch.tensor([torch.exp(torch.tensor(-((x - window_size // 2) ** 2) / (2.0 * 1.5 ** 2))) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        _1d_window = gauss.unsqueeze(1)
        _2d_window = _1d_window @ _1d_window.t()
        window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
        window = window / window.sum()
        return window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if img1.shape[1] != self.channel:
            window = self._create_window(self.window_size, img1.shape[1]).to(img1.device)
        else:
            window = self.window

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=img2.shape[1])

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=img1.shape[1]) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        loss = 1 - ssim_map

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def gradient(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dh = torch.abs(x[..., 1:, :] - x[..., :-1, :])
        dw = torch.abs(x[..., :, 1:] - x[..., :, :-1])
        return dh, dw

    pred_dh, pred_dw = gradient(pred)
    tgt_dh, tgt_dw = gradient(target)
    return (pred_dh - tgt_dh).abs().mean() + (pred_dw - tgt_dw).abs().mean()


def reflection_consistency_loss(pred_reflection: torch.Tensor, reflection_map: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_reflection, reflection_map)
