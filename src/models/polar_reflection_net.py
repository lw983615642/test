from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.block(x)
        return self.activation(out + residual)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, stride=2),
            CBAMBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class IlluminationEnhancer(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.entry = ConvBlock(1, channels)
        self.down1 = DownsampleBlock(channels, channels * 2)
        self.down2 = DownsampleBlock(channels * 2, channels * 4)
        self.bottleneck = ConvBlock(channels * 4, channels * 4)
        self.up2 = UpsampleBlock(channels * 4, channels * 2)
        self.up1 = UpsampleBlock(channels * 2, channels)
        self.out_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s1 = self.entry(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        b = self.bottleneck(s3)
        u2 = self.up2(b, s2)
        u1 = self.up1(u2, s1)
        illumination = torch.sigmoid(self.out_conv(u1)) * 2.0
        enhanced = x * illumination
        return enhanced, illumination


class ReflectionSuppressor(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 48) -> None:
        super().__init__()
        self.entry = ConvBlock(in_channels, base_channels)
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownsampleBlock(base_channels * 4, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.up3 = UpsampleBlock(base_channels * 4, base_channels * 4)
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.up1 = UpsampleBlock(base_channels * 2, base_channels)
        self.out_clean = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.out_reflection = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, enhanced: torch.Tensor, reflection: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([enhanced, reflection], dim=1)
        s1 = self.entry(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b = self.bottleneck(s4)
        u3 = self.up3(b, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)
        clean = torch.sigmoid(self.out_clean(u1))
        predicted_reflection = torch.sigmoid(self.out_reflection(u1))
        return clean, predicted_reflection


class PolarReflectionRemovalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.illumination = IlluminationEnhancer()
        self.suppressor = ReflectionSuppressor()

    def forward(self, input_image: torch.Tensor, reflection_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enhanced, illumination = self.illumination(input_image)
        clean, predicted_reflection = self.suppressor(enhanced, reflection_map)
        return clean, enhanced, predicted_reflection
