# Ablations (validation set, reduced CIFAR-10)

Baseline: validation accuracy 50.64 ± 0.55 %, ECE 42.57 ± 0.66 %.

## Dropout rate above the grid (dense)

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| selected rate | 55.46 ± 0.60 | - | +4.83 | 44.54 ± 0.60 | 37.12 ± 0.48 |
| p=0.8 | 55.82 ± 1.01 | +0.35 pp (no) | +5.18 | 44.18 ± 1.01 | 36.44 ± 1.00 |
| p=0.9 | 54.12 ± 1.20 | -1.34 pp (yes) | +3.49 | 45.85 ± 1.20 | 35.69 ± 0.68 |

## Dropout position: conv layers at their own rates

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| dense (selected) | 55.46 ± 0.60 | - | +4.83 | 44.54 ± 0.60 | 37.12 ± 0.48 |
| conv p=0.1 | 55.19 ± 1.09 | -0.27 pp (no) | +4.55 | 44.81 ± 1.09 | 37.38 ± 0.84 |
| conv p=0.2 | 56.43 ± 1.88 | +0.97 pp (no) | +5.80 | 43.57 ± 1.88 | 34.55 ± 1.40 |
| conv p=0.3 | 57.96 ± 3.45 | +2.50 pp (no) | +7.32 | 42.04 ± 3.45 | 31.51 ± 2.65 |
| conv p=0.4 | 59.95 ± 1.01 | +4.49 pp (yes) | +9.32 | 40.05 ± 1.01 | 26.92 ± 0.65 |
| conv p=0.5 | 59.49 ± 2.23 | +4.03 pp (yes) | +8.85 | 40.22 ± 2.10 | 21.92 ± 1.36 |

## DropBlock block size

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| block 7 (paper) | 53.52 ± 0.55 | - | +2.88 | 46.48 ± 0.55 | 36.69 ± 0.66 |
| block 1 | 53.16 ± 0.81 | -0.36 pp (no) | +2.52 | 46.84 ± 0.81 | 36.91 ± 0.90 |
| block 3 | 54.03 ± 0.37 | +0.52 pp (no) | +3.40 | 45.97 ± 0.37 | 36.44 ± 0.65 |
| block 5 | 53.46 ± 0.56 | -0.06 pp (no) | +2.82 | 46.54 ± 0.56 | 37.02 ± 0.31 |

## DropBlock placement

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| blocks 2-3 (paper) | 53.52 ± 0.55 | - | +2.88 | 46.48 ± 0.55 | 36.69 ± 0.66 |
| block 3 only | 52.74 ± 1.03 | -0.77 pp (no) | +2.10 | 47.26 ± 1.03 | 39.02 ± 1.11 |
| blocks 1-3 | 53.83 ± 1.00 | +0.31 pp (no) | +3.19 | 46.17 ± 1.00 | 34.74 ± 0.66 |

## DropBlock rate schedule

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| linear (paper) | 53.52 ± 0.55 | - | +2.88 | 46.48 ± 0.55 | 36.69 ± 0.66 |
| constant | 55.60 ± 1.10 | +2.09 pp (yes) | +4.97 | 44.40 ± 1.10 | 34.57 ± 0.88 |

Mean ± sample std over seeds, last-epoch models, validation set (the test set is not used). Conclusive when |Δ| > max(std of the variant, std of the selected configuration). The selected configurations were chosen on this validation set with seed 0, so their score is slightly optimistic.
