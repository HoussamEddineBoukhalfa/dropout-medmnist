"""The common CNN (section 3 of the brief) with optional dropout variants.

    3 x [conv3x3 - ReLU - conv3x3 - ReLU - (DropBlock / spatial dropout) - maxpool2 - (dropout, conv position)]
    with 32, 64, 128 filters, then dense(256) - ReLU - (dropout, dense position) - dense(n_classes).

With every rate at 0 this is the unregularised baseline: no dropout, no normalisation layers.
Convolutions use padding 1, so 28x28 inputs give 128 x 3 x 3 features before the dense layer.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .layers import DropBlock2d

STOCHASTIC_LAYERS = (nn.Dropout, nn.Dropout2d, DropBlock2d)


class SmallCNN(nn.Module):
    def __init__(self, cfg: ModelConfig, in_channels: int, n_classes: int, image_size: int = 28):
        super().__init__()
        if cfg.dropout_pos not in ("dense", "conv"):
            raise ValueError(f"dropout_pos must be 'dense' or 'conv', got '{cfg.dropout_pos}'")
        blocks = []
        channels = in_channels
        for i, width in enumerate(cfg.widths, start=1):
            layers: list[nn.Module] = [
                nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            ]
            if cfg.dropblock_p > 0 and i in cfg.dropblock_blocks:
                layers.append(DropBlock2d(cfg.dropblock_p, cfg.dropblock_size, cfg.dropblock_schedule))
            if cfg.spatial_p > 0:
                layers.append(nn.Dropout2d(cfg.spatial_p))
            layers.append(nn.MaxPool2d(2))
            if cfg.dropout_p > 0 and cfg.dropout_pos == "conv":
                layers.append(nn.Dropout(cfg.dropout_p))
            blocks.append(nn.Sequential(*layers))
            channels = width
        self.features = nn.Sequential(*blocks)

        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, in_channels, image_size, image_size)).numel()
        head: list[nn.Module] = [nn.Flatten(), nn.Linear(n_flat, cfg.hidden), nn.ReLU(inplace=True)]
        if cfg.dropout_p > 0 and cfg.dropout_pos == "dense":
            head.append(nn.Dropout(cfg.dropout_p))
        head.append(nn.Linear(cfg.hidden, n_classes))
        self.classifier = nn.Sequential(*head)

        if cfg.init == "kaiming":
            self.apply(_he_init)
        elif cfg.init != "default":
            raise ValueError(f"init must be 'kaiming' or 'default', got '{cfg.init}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def has_stochastic_layers(self) -> bool:
        return any(isinstance(m, STOCHASTIC_LAYERS) for m in self.modules())


def _he_init(module: nn.Module) -> None:
    """He (Kaiming) normal initialisation, as in task A's PathMNIST notebook. Without it the
    unregularised network diverges at lr 0.05 on PathMNIST; it is not a regulariser."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)


def build_model(cfg: ModelConfig, in_channels: int, n_classes: int, image_size: int) -> SmallCNN:
    return SmallCNN(cfg, in_channels, n_classes, image_size)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)