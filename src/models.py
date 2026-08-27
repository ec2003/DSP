"""Speaker embedding model: log-mel CNN encoder trained with an ArcFace head.

The encoder is trained from scratch rather than fine-tuned from a large
pretrained speaker model. Pretrained speaker encoders are typically trained
with heavy noise augmentation and are therefore already noise-robust, which
would mask the effect of the DSP front-end this study measures.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.features import LogMelSpectrogram


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SpeakerCNN(nn.Module):
    """Log-mel CNN with attentive statistics pooling producing L2-normalised embeddings."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        embedding_dim: int = 192,
    ) -> None:
        super().__init__()
        self.frontend = LogMelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(_ConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        pooled_mels = n_mels // (2 ** len(channels))
        if pooled_mels < 1:
            raise ValueError(
                f"{len(channels)} pooling stages is too many for {n_mels} mel bins"
            )
        self.feature_dim = channels[-1] * pooled_mels

        self.attention = nn.Sequential(
            nn.Conv1d(self.feature_dim, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, self.feature_dim, kernel_size=1),
        )
        self.embedding = nn.Sequential(
            nn.Linear(self.feature_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.frontend(waveform).unsqueeze(1)
        features = self.blocks(features)
        batch, channels, mels, frames = features.shape
        features = features.reshape(batch, channels * mels, frames)

        weights = torch.softmax(self.attention(features), dim=-1)
        mean = torch.sum(features * weights, dim=-1)
        variance = torch.sum((features**2) * weights, dim=-1) - mean**2
        pooled = torch.cat([mean, variance.clamp_min(1e-8).sqrt()], dim=1)
        return F.normalize(self.embedding(pooled), p=2, dim=1)


class ArcFaceHead(nn.Module):
    """Additive angular margin classification head used only during training."""

    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(embeddings, F.normalize(self.weight, p=2, dim=1)).clamp(
            -1 + 1e-7, 1 - 1e-7
        )
        theta = torch.acos(cosine)
        target = F.one_hot(labels, num_classes=self.weight.shape[0]).bool()
        # Angular margin is applied only where it keeps theta inside [0, pi].
        margined = torch.where(
            theta + self.margin < math.pi, theta + self.margin, theta
        )
        logits = torch.where(target, torch.cos(margined), cosine)
        return logits * self.scale
