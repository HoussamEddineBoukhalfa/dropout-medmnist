# Task C - reduced CIFAR-10 control

Part of the G2 Journal Club project (dropout and uncertainty). This folder trains the unregularised
baseline, standard dropout and DropBlock on **reduced CIFAR-10 with exactly the PathMNIST protocol**.

## Why

Our PathMNIST setup differs from Opperman et al. (2026): other architecture, 4,500 training images,
100 epochs, last-epoch model. If dropout behaves differently on PathMNIST than in the paper, the cause
may be the **domain** (medical images) or **our protocol**. The same protocol on non-medical images
separates the two:

| Effect on CIFAR-10 (this task) | Effect on PathMNIST (tasks A, B) | Reading |
|---|---|---|
| helps | helps | consistent with the paper, no domain effect detected |
| helps | does not help | **domain effect**: the conclusion does not transfer to histology |
| does not help | does not help | **protocol effect**: our setup, not the images, explains the gap with the paper |
| does not help | helps | domain effect in the other direction; check the variance first |

"Helps" means a test-accuracy difference to the baseline larger than the seed standard deviation
(the brief's rule), shown as *conclusive* in `summary.md`.

## Protocol

Identical to PathMNIST:

* Network: 3 x [conv3x3-ReLU-conv3x3-ReLU-maxpool], 32/64/128 filters, dense 256, 28x28 inputs, no normalisation layers.
* Training: cross-entropy, SGD momentum 0.9, lr 0.05 with cosine decay (per epoch), batch 128, 100 epochs, last-epoch model.
* Data regime: balanced 4,500-image training subset (fixed seed), 8,000 validation images, no augmentation.
* Preprocessing: per-channel standardisation with training-subset statistics only.
* Tuning: one hyperparameter per method, at most 5 values, 1 seed, last-epoch validation accuracy.
* Final: 3 seeds, mean ± sample std, Δ vs baseline; accuracy, generalisation gap, ECE (15 bins), NLL, macro AUC.

Necessarily different:

| | PathMNIST | CIFAR-10 here | Why it is acceptable |
|---|---|---|---|
| Classes | 9 | 10 (450 training images each) | only the output layer changes |
| Image size | 28x28 | 32x32 resized once to 28x28 | keeps the network identical; fixed preprocessing, not augmentation |
| Validation | 8,000 from the official val split | 8,000 from training images not used for training | CIFAR-10 has no val split; same size, disjoint |
| Test | 2,000 same-centre + 7,180 other-centre | official test set, 10,000 | no centre shift in CIFAR-10: only the overfitting comparison applies |

## Plan

| Stage | Runs |
|---|---|
| Baseline, seed 0 + overfitting check | 1 |
| Dropout rate at the dense position, grid {0.1, 0.2, 0.3, 0.5, 0.7} | 5 |
| Dropout at the conv position, selected rate | 1 |
| DropBlock rate (block 3, blocks 2-3), grid {0.05, 0.1, 0.15, 0.2, 0.3} | 5 |
| Seeds 1 and 2 for baseline, dropout, DropBlock (seed 0 reused) | 6 |
| **Total** | **18 trainings** |

Choosing the dropout position after the rate keeps one tuned hyperparameter per method.
With `--mc-samples T`, the selected dropout and DropBlock models are also evaluated with Monte Carlo
dropout (T stochastic passes), to compare with task D on PathMNIST.

## Layout

```
task_c/
  configs/         protocol.yaml (shared values), baseline.yaml, dropout.yaml, dropblock.yaml
  src/cifar_control/
    config.py      YAML configs with inheritance and --set overrides, run ids
    data.py        reduced CIFAR-10 splits and preprocessing
    layers.py      DropBlock
    models.py      the protocol CNN with optional dropout / DropBlock
    train.py       training loop
    runs.py        run folders, caching, sweeps, selection on validation, overfitting check
    metrics.py     accuracy, NLL, ECE, macro AUC
    evaluate.py    test evaluation (once), MC dropout
    reporting.py   summary table and figures
    pipeline.py    the stages
  scripts/
    run.py         everything, end to end
    report.py      rebuild the table and figures from saved results
  tests/           pytest, runs on CPU with synthetic data
  results/         committed: sweeps, selected configs, per-seed metrics, summary.md, figures
  outputs/         not committed: checkpoints and histories
```

## Running

```bash
cd task_c
pip install -e ".[dev]"
pytest                          # under a minute on CPU
python scripts/run.py --quick   # 2-epoch dry run of every stage (writes to results/quick/)
python scripts/run.py           # full protocol
```

On Google Colab with a GPU runtime, keep runs and results on Drive so a disconnect loses nothing.
Running the same command again resumes: finished runs are never retrained.

```bash
cd task_c && pip install -q -e .
python scripts/run.py --out /content/drive/MyDrive/task_c/outputs \
                      --results /content/drive/MyDrive/task_c/results --mc-samples 20
```

Then copy the `results/` folder from Drive into `task_c/results/` and commit it.

## Outputs

`results/cifar10/`:

* `summary.md` - the results table for the report; `summary.csv` - every metric, mean and std
* `final/per_seed.csv` - test metrics for each method and seed
* `sweeps/*.csv`, `selected/*.yaml` - tuning results and the chosen configurations
* `curves/` - baseline learning curves and the overfitting check
* `figures/` - learning curves, tuning sweeps, final metrics, reliability diagrams

The test set is evaluated once and cached per run; `--force-eval` recomputes it.

## To align with tasks A and B before the final runs

* The dropout grid (A) and the DropBlock block size, blocks and grid (B), in `configs/`.
* The split seed (42) and the normalisation statistics (training subset), in `configs/protocol.yaml`.
* The network: as written in the brief it has 0.58 M parameters, while the brief says "about 1 M".
  Check that A's PathMNIST model is the same; if not, copy A's definition into `models.py` so the
  control uses the same network.
