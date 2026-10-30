"""DropBlock (Ghiasi, Lin & Le, 2018).

Dropout on convolutional feature maps is weak because neighbouring activations are correlated:
a zeroed unit can be recovered from its neighbours. DropBlock zeroes contiguous
``block_size x block_size`` regions instead.

Algorithm 1 of the paper:

1. sample block centres with Bernoulli(gamma) inside the region where a full block fits,
   gamma = p / block_size^2 * (H * W) / ((H - block_size + 1) * (W - block_size + 1)),
   so that roughly a fraction ``p`` of units is dropped;
2. expand every centre to a block (max-pooling of the centre mask);
3. multiply by the mask and rescale by count(mask) / count(ones) to keep the expected activation.

Masks are sampled independently per sample and per channel. Like dropout, DropBlock is the
identity in ``eval()`` mode. Special cases: ``block_size = 1`` is standard dropout; a block
covering the whole map behaves like spatial dropout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DropBlock2d(nn.Module):
    def __init__(self, p: float, block_size: int):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        if block_size < 1 or block_size % 2 == 0:
            raise ValueError(f"block_size must be a positive odd integer, got {block_size}")
        self.p = p
        self.block_size = block_size

    def extra_repr(self) -> str:
        return f"p={self.p}, block_size={self.block_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        n, c, h, w = x.shape
        bs = min(self.block_size, h, w)
        if bs % 2 == 0:  # map smaller than the block: largest odd block that fits
            bs -= 1
        valid_h, valid_w = h - bs + 1, w - bs + 1
        gamma = self.p / bs ** 2 * (h * w) / (valid_h * valid_w)

        centres = torch.bernoulli(torch.full((n, c, valid_h, valid_w), min(gamma, 1.0),
                                             device=x.device, dtype=x.dtype))
        pad = bs // 2
        centres = F.pad(centres, (pad, pad, pad, pad))
        dropped = F.max_pool2d(centres, kernel_size=bs, stride=1, padding=pad)
        mask = 1.0 - dropped
        keep = mask.sum()
        if keep == 0:
            return torch.zeros_like(x)
        return x * mask * (mask.numel() / keep)
