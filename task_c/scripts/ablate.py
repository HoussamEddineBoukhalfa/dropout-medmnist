#!/usr/bin/env python
"""Ablations around the selected dropout (dense, p=0.7) and DropBlock (p=0.2, block 3) configurations.

Variants are defined in configs/ablations.yaml. Measured on the validation set with 3 seeds; the test
set is not used. Finished runs are reused (the selected configurations and the baseline are already
trained if run.py ran in the same output folder). Writes results/<dataset>/ablations/.

  python scripts/ablate.py --quick --set data.root=/tmp/cifar     # 2-epoch dry run, 1 seed
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
    references = load_references(spec, configs, overrides)
    baseline = load_config(configs / "baseline.yaml", overrides)
    results = results_root / baseline.data.name

    # the references must be the configurations run.py selected
    for name, ref in references.items():
        selected = results / "selected" / f"{name}.yaml"
        if selected.exists() and load_config(selected, overrides).run_id != ref.run_id:
            log.warning("reference '%s' in %s differs from %s: update the spec if the selection changed",
                        name, args.spec, selected)

    variants = build_variants(spec, references)
    log.info("%d configurations x %d seeds (finished runs are reused)", len({v.cfg.run_id for v in variants}) + 1,
             len(seeds))
    per_seed = run_ablations(variants, baseline, seeds, out, device)
    write_report(per_seed, spec, results / "ablations")
    print((results / "ablations" / "ablations.md").read_text())


if __name__ == "__main__":
    main()
