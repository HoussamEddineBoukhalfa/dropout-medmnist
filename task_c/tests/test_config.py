from __future__ import annotations

from pathlib import Path

import pytest

from cifar_control.config import from_dict, load_config, parse_overrides, with_param

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_configs_follow_the_protocol():
    base = load_config(CONFIGS / "baseline.yaml")
    dropout = load_config(CONFIGS / "dropout.yaml")
    dropblock = load_config(CONFIGS / "dropblock.yaml")
    assert base.data.name == "cifar10" and base.data.n_train == 4500 and base.data.n_val == 8000
    assert base.train.epochs == 100 and base.train.lr == 0.05 and base.train.batch_size == 128
    assert base.model.dropout_p == 0.0 and base.model.dropblock_p == 0.0 and base.train.weight_decay == 0.0
    for cfg in (dropout, dropblock):
        assert len(cfg.sweep.values) <= 5  # at most 5 values per hyperparameter


def test_baseline_and_zero_rate_share_a_run_id():
    base = load_config(CONFIGS / "baseline.yaml")
    dropout = load_config(CONFIGS / "dropout.yaml")
    assert base.run_id == dropout.run_id  # name and sweep spec are not part of the identity
    assert with_param(dropout, "model.dropout_p", 0.3).run_id != base.run_id


def test_int_and_float_values_hash_identically():
    assert from_dict({"model": {"dropout_p": 0}}).run_id == from_dict({"model": {"dropout_p": 0.0}}).run_id


def test_data_root_is_not_part_of_the_identity():
    assert from_dict({"data": {"root": "a"}}).run_id == from_dict({"data": {"root": "b"}}).run_id


def test_overrides_are_typed():
    assert parse_overrides(["train.epochs=2", "model.dropout_pos=conv"]) == {
        "train": {"epochs": 2}, "model": {"dropout_pos": "conv"}}
    assert load_config(CONFIGS / "baseline.yaml", ["train.epochs=2"]).train.epochs == 2


def test_unknown_keys_are_rejected():
    with pytest.raises(KeyError):
        from_dict({"model": {"dropuot_p": 0.1}})
    with pytest.raises(KeyError):
        with_param(from_dict({}), "model.nope", 1)
