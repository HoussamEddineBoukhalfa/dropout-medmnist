"""Experiment configuration.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    name: str = "cifar10"  # cifar10 | synthetic (offline tests)
    root: str = "data"  # download / cache directory (not part of the run id)
    split_seed: int = 42  # fixes the training subset and the validation draw
    n_train: int = 4500  # balanced training subset: 450 images per class
    n_val: int = 8000  # validation images, used for tuning only
    image_size: int = 28  # CIFAR-10 is resized 32 -> 28 so the network matches PathMNIST


@dataclass
class ModelConfig:
    widths: list[int] = field(default_factory=lambda: [32, 64, 128])
    hidden: int = 256
    dropout_p: float = 0.0  # standard element-wise dropout rate
    dropout_pos: str = "dense"  # dense: after the hidden dense layer | conv: after each pooling layer
    spatial_p: float = 0.0  # spatial dropout (whole feature maps) before each pooling layer
    dropblock_p: float = 0.0  # DropBlock drop rate
    dropblock_size: int = 3  # DropBlock block side (odd)
    dropblock_blocks: list[int] = field(default_factory=lambda: [2, 3])  # conv blocks (1-indexed) with DropBlock


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler: str = "cosine"  # cosine | constant


@dataclass
class SweepConfig:
    param: str  # dotted key, e.g. model.dropout_p
    values: list[Any]


@dataclass
class ExperimentConfig:
    name: str
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sweep: SweepConfig | None = None

    def identity(self) -> dict[str, Any]:
        """Fields that change the trained model. Name, sweep spec and data.root are excluded."""
        data = asdict(self.data)
        data.pop("root")
        return {"data": data, "model": asdict(self.model), "train": asdict(self.train)}

    @property
    def run_id(self) -> str:
        blob = json.dumps(self.identity(), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.sweep is None:
            out.pop("sweep")
        return out


_SECTIONS = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig}


def _coerce(type_name: str, value: Any) -> Any:
    """Coerce YAML values to the declared type so that 0 and 0.0 hash to the same run id."""
    if value is None:
        return None
    if type_name == "float":
        return float(value)
    if type_name == "int":
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"expected an integer, got {value}")
        return int(value)
    if type_name == "str":
        return str(value)
    if type_name == "list[int]":
        return [int(v) for v in value]
    return value


def _build(cls: type, values: dict[str, Any], where: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise KeyError(f"unknown key(s) in '{where}': {unknown}")
    return cls(**{k: _coerce(str(known[k].type), v) for k, v in values.items()})


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    if parent:
        raw = _deep_merge(_read_yaml((path.parent / parent).resolve()), raw)
    return raw


def parse_overrides(items: list[str] | None) -> dict:
    """``["train.epochs=2", "model.dropout_pos=conv"]`` -> nested dict with YAML-typed values."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got '{item}'")
        key, raw_value = item.split("=", 1)
        node = out
        *parents, leaf = key.strip().split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw_value)
    return out


def from_dict(raw: dict, default_name: str = "experiment") -> ExperimentConfig:
    raw = copy.deepcopy(raw)
    name = raw.pop("name", default_name)
    sections = {k: _build(cls, raw.pop(k, None) or {}, k) for k, cls in _SECTIONS.items()}
    sweep = raw.pop("sweep", None)
    if raw:
        raise KeyError(f"unknown top-level key(s): {sorted(raw)}")
    return ExperimentConfig(name=name, sweep=SweepConfig(**sweep) if sweep else None, **sections)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    path = Path(path)
    raw = _read_yaml(path)
    if overrides:
        raw = _deep_merge(raw, parse_overrides(overrides))
    return from_dict(raw, default_name=path.stem)


def with_param(cfg: ExperimentConfig, dotted: str, value: Any, name: str | None = None) -> ExperimentConfig:
    """Copy of ``cfg`` with one field changed (e.g. ``model.dropout_p``) and no sweep spec."""
    section, key = dotted.split(".", 1)
    if section not in _SECTIONS:
        raise KeyError(f"'{dotted}': section must be one of {list(_SECTIONS)}")
    raw = cfg.to_dict()
    raw.pop("sweep", None)
    if key not in raw[section]:
        raise KeyError(f"'{dotted}': unknown key '{key}' in section '{section}'")
    raw[section][key] = value
    raw["name"] = name or cfg.name
    return from_dict(raw)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
