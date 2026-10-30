from __future__ import annotations

import pytest
import torch

from cifar_control.layers import DropBlock2d


def test_identity_in_eval_mode():
    layer = DropBlock2d(0.3, 3).eval()
    x = torch.randn(2, 4, 14, 14)
    assert torch.equal(layer(x), x)


@pytest.mark.parametrize("p, size", [(0.1, 3), (0.2, 3), (0.3, 5)])
def test_drop_rate_and_rescaling(p, size):
    torch.manual_seed(0)
    layer = DropBlock2d(p, size).train()
    out = layer(torch.ones(64, 16, 14, 14))
    dropped = (out == 0).float().mean().item()
    assert abs(dropped - p) < 0.06  # overlapping blocks make the rate slightly lower than p
    assert abs(out.mean().item() - 1.0) < 1e-4  # expectation preserved


def test_drops_contiguous_blocks():
    torch.manual_seed(1)
    out = DropBlock2d(0.2, 3).train()(torch.ones(1, 1, 14, 14))[0, 0]
    zeros = (out == 0).nonzero().tolist()
    assert zeros, "nothing dropped"
    r, c = zeros[0]
    assert (out[r:r + 3, c:c + 3] == 0).all()  # top-left zero starts a full 3x3 block


def test_rejects_even_block_size():
    with pytest.raises(ValueError):
        DropBlock2d(0.1, 4)
