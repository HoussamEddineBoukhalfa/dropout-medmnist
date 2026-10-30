"""arguments shared by the two scripts. Default paths are relative to the task_c folder,
so the scripts work from any working directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .utils import get_device, setup_logging

PROJECT = Path(__file__).resolve().parents[2]


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", default=str(PROJECT / "outputs"),
                        help="runs: checkpoints and histories (not committed). On Colab, point it to Google Drive")
    parser.add_argument("--results", default=str(PROJECT / "results"),
                        help="tables, selected configs and figures (committed)")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. train.epochs=2 data.root=/content/data")
    parser.add_argument("--log-level", default="INFO")
    return parser


def setup(args: argparse.Namespace) -> torch.device:
    setup_logging(args.log_level)
    return get_device(args.device)
