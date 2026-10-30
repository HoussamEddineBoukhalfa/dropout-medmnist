from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cifar_control.ablation import apply_changes, build_variants, load_references, load_spec, run_ablations, write_report
from cifar_control.config import load_config, save_config, with_param

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CPU = torch.device("cpu")


@pytest.fixture
def selected_dir(tmp_path) -> Path:
    """What run.py writes to results/<dataset>/selected/."""
    out = tmp_path / "selected"
    save_config(with_param(load_config(CONFIGS / "dropout.yaml"), "model.dropout_p", 0.7), out / "dropout.yaml")
    save_config(with_param(load_config(CONFIGS / "dropblock.yaml"), "model.dropblock_p", 0.2), out / "dropblock.yaml")
    return out


def test_references_are_the_selected_configurations(selected_dir):
    refs = load_references(load_spec(CONFIGS / "ablations.yaml"), CONFIGS, selected_dir=selected_dir)
    assert refs["dropout"].model.dropout_p == 0.7 and refs["dropout"].model.dropout_pos == "dense"
    db = refs["dropblock"].model
    assert (db.dropblock_p, db.dropblock_size, db.dropblock_schedule) == (0.2, 7, "linear")
    assert refs["dropout"].run_id == load_config(selected_dir / "dropout.yaml").run_id


def test_missing_selection_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="run scripts/run.py first"):
        load_references(load_spec(CONFIGS / "ablations.yaml"), CONFIGS, selected_dir=tmp_path)


def test_each_variant_changes_only_what_it_says(selected_dir):
    spec = load_spec(CONFIGS / "ablations.yaml")
    variants = {(v.group, v.label): v.cfg for v in build_variants(spec, load_references(spec, CONFIGS,
                                                                                         selected_dir=selected_dir))}
    assert variants[("dropblock_size", "block 3")].model.dropblock_size == 3
    assert variants[("dropblock_size", "block 3")].model.dropblock_p == 0.2
    assert variants[("dropblock_placement", "block 3 only")].model.dropblock_blocks == [3]
    assert variants[("dropblock_schedule", "constant")].model.dropblock_schedule == "constant"
    conv = variants[("dropout_position", "conv p=0.4")].model
    assert (conv.dropout_pos, conv.dropout_p) == ("conv", 0.4)
    # 12 variants + 2 references (each reference appears in several groups)
    assert len({cfg.run_id for cfg in variants.values()}) == 14


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
    assert not variant["reference"]
    assert not list((tmp_path / "out").rglob("test_metrics.json"))  # the test set was never loaded