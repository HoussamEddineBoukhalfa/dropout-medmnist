#!/usr/bin/env python
"""Task C - reduced CIFAR-10 control, end to end.

Stages (finished runs are cached, so re-running resumes after a disconnect):

  1. baseline, tuning seed       -> learning curves + overfitting check
  2. dropout rate sweep (dense)  -> validation only, 1 seed
  3. dropout position            -> conv vs dense at the selected rate, 1 seed
  4. DropBlock rate sweep        -> validation only, 1 seed
  5. final runs                  -> baseline + selected dropout + selected DropBlock, 3 seeds
  6. test evaluation             -> once, at the end
  7. report                      -> results/<dataset>/summary.md and figures/

Examples
--------
  python scripts/run.py --quick                                 # 2-epoch dry run of every stage
  python scripts/run.py                                         # full protocol
  python scripts/run.py --out /content/drive/MyDrive/task_c/outputs \\
                        --results /content/drive/MyDrive/task_c/results --mc-samples 20
  python scripts/run.py --quick --set data.name=synthetic       # offline smoke test
"""

from __future__ import annotations

from pathlib import Path

from cifar_control.cli import PROJECT, base_parser, setup
from cifar_control.config import load_config
from cifar_control.pipeline import (baseline_stage, build_report, evaluate_stage, final_stage, tune_stage)


def main() -> None:
    parser = base_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--configs", default=str(PROJECT / "configs"),
                        help="folder with baseline.yaml, dropout.yaml, dropblock.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="final seeds; the first one is also the tuning seed")
    parser.add_argument("--mc-samples", type=int, default=0,
                        help="also report Monte Carlo dropout with this many passes (0 = off)")
    parser.add_argument("--force-eval", action="store_true", help="recompute test metrics even if cached")
    parser.add_argument("--quick", action="store_true",
                        help="dry run: 2 epochs, first 2 grid values, separate output folders")
    args = parser.parse_args()
    device = setup(args)

    overrides = [f"data.root={PROJECT / 'data'}", *args.set]
    out, results_root = Path(args.out), Path(args.results)
    if args.quick:
        overrides.append("train.epochs=2")
        out, results_root = out / "quick", results_root / "quick"

    configs = Path(args.configs)
    baseline = load_config(configs / "baseline.yaml", overrides)
    dropout = load_config(configs / "dropout.yaml", overrides)
    dropblock = load_config(configs / "dropblock.yaml", overrides)
    results = results_root / baseline.data.name
    seed = args.seeds[0]

    def grid(cfg):
        return cfg.sweep.values[:2] if args.quick else cfg.sweep.values

    # 1. baseline + overfitting check
    baseline_stage(baseline, seed, out, results, device)

    # 2-3. dropout: rate at the dense position, then position at the selected rate
    dropout_rate = tune_stage(dropout, dropout.sweep.param, grid(dropout), seed, out, results, device,
                              sweep_name="dropout_rate", selected_name="dropout_rate")
    dropout_final = tune_stage(dropout_rate, "model.dropout_pos", ["dense", "conv"], seed, out, results, device,
                               sweep_name="dropout_position", selected_name="dropout")

    # 4. DropBlock rate
    dropblock_final = tune_stage(dropblock, dropblock.sweep.param, grid(dropblock), seed, out, results, device,
                                 sweep_name="dropblock_rate", selected_name="dropblock")

    # 5. final runs, 3 seeds (the tuning-seed runs are reused from the cache)
    methods = {"baseline": baseline, "dropout": dropout_final, "dropblock": dropblock_final}
    final_stage(methods, args.seeds, out, device)

    # 6. test set, used once
    evaluate_stage(methods, args.seeds, out, results, device, mc_samples=args.mc_samples, force=args.force_eval)

    # 7. table and figures
    build_report(results, out)


if __name__ == "__main__":
    main()
