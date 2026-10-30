from __future__ import annotations

import pytest
import torch
from torch import nn

from cifar_control.config import ModelConfig
from cifar_control.layers import DropBlock2d
from cifar_control.models import build_model, count_parameters


def _layers(model, kind):
    return [m for m in model.modules() if isinstance(m, kind)]


def test_baseline_has_no_regularisation_layers():
    model = build_model(ModelConfig(), 3, 10, 28)
    assert not model.has_stochastic_layers()
    assert not _layers(model, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm))
    assert model(torch.randn(2, 3, 28, 28)).shape == (2, 10)
    assert 500_000 < count_parameters(model) < 1_200_000


@pytest.mark.parametrize("pos, expected", [("dense", 1), ("conv", 3)])
def test_dropout_position(pos, expected):
    model = build_model(ModelConfig(dropout_p=0.3, dropout_pos=pos), 3, 10, 28)
    assert len(_layers(model, nn.Dropout)) == expected


def test_dropblock_only_in_selected_blocks():
    model = build_model(ModelConfig(dropblock_p=0.1, dropblock_blocks=[2, 3]), 3, 10, 28)
    assert len(_layers(model, DropBlock2d)) == 2
    assert not _layers(model.features[0], DropBlock2d)


def test_he_init_matches_task_a():
    torch.manual_seed(0)
    model = build_model(ModelConfig(init="kaiming"), 3, 10, 28)
    layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert all(torch.count_nonzero(m.bias) == 0 for m in layers)  # zero biases
    conv = layers[1]  # 32 -> 32 channels, 3x3: He std = sqrt(2 / fan_in)
    expected = (2.0 / (32 * 3 * 3)) ** 0.5
    assert abs(conv.weight.std().item() - expected) / expected < 0.1
    default = build_model(ModelConfig(init="default"), 3, 10, 28)
    assert torch.count_nonzero(next(m for m in default.modules() if isinstance(m, nn.Conv2d)).bias) > 0
    with pytest.raises(ValueError):
        build_model(ModelConfig(init="xavier"), 3, 10, 28)


def test_eval_mode_is_deterministic():
    model = build_model(ModelConfig(dropout_p=0.5, dropblock_p=0.2), 3, 9, 28).eval()
    x = torch.randn(4, 3, 28, 28)
    assert torch.equal(model(x), model(x))