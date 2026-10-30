#!/usr/bin/env python
"""Ablations around the dropout and DropBlock configurations selected by scripts/run.py.

Run scripts/run.py first with the same --out / --results: the references are read from
results/<dataset>/selected/. Variants are defined in configs/ablations.yaml. Measured on the
validation set with 3 seeds; the test set is not used. Finished runs are reused.
Writes results/<dataset>/ablations/.

  python scripts/ablate.py --quick --set data.root=/tmp/cifar     # after run.py --quick
  python scripts/ablate.py --set data.root=/tmp/cifar
"""

from __future__ import annotations

import logging
from pathlib import Path

from cifar_control.ablation import build_variants, load_references, load_spec, run_ablations, write_report
from cifar_control.cli import PROJECT, base_parser, setup
from cifar_control.config import load_config

log = logging.getLogger("ablate")


def main() -> None:
    parser = base_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--spec", default=str(PROJECT / "configs" / "ablations.yaml"))
    parser.add_argument("--configs", default=str(PROJECT / "configs"))
    parser.add_argument("--seeds", type=int, nargs="+", help="default: the seeds listed in the spec")
    parser.add_argument("--quick", action="store_true", help="dry run: 2 epochs, 1 seed, separate folders")
    args = parser.parse_args()
    device = setup(args)

    overrides = [f"data.root={PROJECT / 'data'}", *args.set]
    out, results_root = Path(args.out), Path(args.results)
    spec = load_spec(args.spec)
    seeds = args.seeds or spec["seeds"]
    if args.quick:
        overrides.append("train.epochs=2")
        out, results_root, seeds = out / "quick", results_root / "quick", seeds[:1]

    configs = Path(args.configs)
    baseline = load_config(configs / "baseline.yaml", overrides)
    results = results_root / baseline.data.name
    references = load_references(spec, configs, overrides, selected_dir=results / "selected")
    for name, ref in references.items():
        log.info("reference %s: %s (run %s)", name,
                 {k: v for k, v in vars(ref.model).items() if k.startswith(("dropout", "dropblock"))}, ref.run_id)

    variants = build_variants(spec, references)
    log.info("%d configurations x %d seeds (finished runs are reused)", len({v.cfg.run_id for v in variants}) + 1,
             len(seeds))
    per_seed = run_ablations(variants, baseline, seeds, out, device)
    write_report(per_seed, spec, results / "ablations")
    print((results / "ablations" / "ablations.md").read_text())


if __name__ == "__main__":
    main()