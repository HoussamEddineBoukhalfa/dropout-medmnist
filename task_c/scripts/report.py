#!/usr/bin/env python
"""Rebuild the summary table and figures from results on disk (no training, no new test evaluation).

  python scripts/report.py
  python scripts/report.py --out /content/drive/MyDrive/task_c/outputs \\
                           --results /content/drive/MyDrive/task_c/results
"""

from __future__ import annotations

from pathlib import Path

from cifar_control.cli import base_parser, setup
from cifar_control.pipeline import build_report


def main() -> None:
    parser = base_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="cifar10")
    args = parser.parse_args()
    setup(args)
    build_report(Path(args.results) / args.dataset, args.out)


if __name__ == "__main__":
    main()
