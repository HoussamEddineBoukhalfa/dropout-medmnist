from __future__ import annotations

from pathlib import Path

import torch

from cifar_control.ablation import (apply_changes, build_variants, load_references, load_spec, run_ablations,
                                    write_report)
from cifar_control.config import with_param

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CPU = torch.device("cpu")


def test_references_match_the_selected_configurations():
    spec = load_spec(CONFIGS / "ablations.yaml")
    refs = load_references(spec, CONFIGS)
    assert refs["dropout"].model.dropout_p == 0.7 and refs["dropout"].model.dropout_pos == "dense"
    assert refs["dropblock"].model.dropblock_p == 0.2 and refs["dropblock"].model.dropblock_size == 3
    # same identity as what run.py's sweep would select (name and sweep spec do not matter)
    from cifar_control.config import load_config
    selected = with_param(load_config(CONFIGS / "dropout.yaml"), "model.dropout_p", 0.7)
    assert refs["dropout"].run_id == selected.run_id


def test_each_variant_changes_only_what_it_says():
    spec = load_spec(CONFIGS / "ablations.yaml")
    refs = load_references(spec, CONFIGS)
    variants = {(v.group, v.label): v.cfg for v in build_variants(spec, refs)}
    assert variants[("dropblock_size", "block 7")].model.dropblock_size == 7
    assert variants[("dropblock_size", "block 7")].model.dropblock_p == 0.2
    assert variants[("dropblock_placement", "block 3 only")].model.dropblock_blocks == [3]
    combo = variants[("combination", "+ DropBlock p=0.2")].model
    assert combo.dropout_p == 0.7 and combo.dropblock_p == 0.2
    # 13 variants + 2 references; the dropout reference sits in 3 groups and the DropBlock one in 2
    assert len({cfg.run_id for cfg in variants.values()}) == 15


def test_ablation_pipeline_on_synthetic(tiny_cfg, tmp_path):
    refs = {"dropout": apply_changes(tiny_cfg, {"model.dropout_p": 0.5}, "dropout"),
            "dropblock": apply_changes(tiny_cfg, {"model.dropblock_p": 0.2}, "dropblock")}
    spec = {"references": {"dropout": {}, "dropblock": {}}, "groups": [
        {"name": "rate", "title": "Rate", "reference": "dropout", "variants": {"p=0.3": {"model.dropout_p": 0.3}}},
        {"name": "size", "title": "Size", "reference": "dropblock",
         "variants": {"block 1": {"model.dropblock_size": 1}}},
    ]}
    per_seed = run_ablations(build_variants(spec, refs), tiny_cfg, [0, 1], tmp_path / "out", CPU)
    assert set(per_seed["group"]) == {"baseline", "rate", "size"} and len(per_seed) == 10
    summary = write_report(per_seed, spec, tmp_path / "ablations")
    assert (tmp_path / "ablations" / "ablations.png").exists()
    assert "## Rate" in (tmp_path / "ablations" / "ablations.md").read_text()
    variant = summary[(summary["group"] == "rate") & (summary["variant"] == "p=0.3")].iloc[0]
    assert abs(variant["delta_vs_reference"]) >= 0 and not variant["reference"]
    # the test set was never loaded: no test metrics written
    assert not list((tmp_path / "out").rglob("test_metrics.json"))