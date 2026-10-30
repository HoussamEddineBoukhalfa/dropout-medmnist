# Ablations (validation set, reduced CIFAR-10)

Baseline: validation accuracy 52.58 ± 0.54 %, ECE 40.73 ± 0.52 %.

## Dropout rate beyond the grid (dense)

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| selected rate | 56.17 ± 0.75 | - | +3.60 | 43.83 ± 0.75 | 36.65 ± 0.60 |
| p=0.9 | 52.50 ± 0.99 | -3.67 pp (yes) | -0.08 | 47.47 ± 0.99 | 37.24 ± 0.88 |

## Dropout position: conv layers at their own rates

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| dense (selected) | 56.17 ± 0.75 | - | +3.60 | 43.83 ± 0.75 | 36.65 ± 0.60 |
| conv p=0.1 | 57.47 ± 0.28 | +1.30 pp (yes) | +4.89 | 42.53 ± 0.28 | 34.76 ± 0.29 |
| conv p=0.2 | 60.30 ± 1.19 | +4.12 pp (yes) | +7.72 | 39.70 ± 1.19 | 29.97 ± 1.06 |
| conv p=0.3 | 61.39 ± 0.93 | +5.22 pp (yes) | +8.81 | 38.60 ± 0.93 | 24.87 ± 0.82 |
| conv p=0.4 | 62.21 ± 0.76 | +6.03 pp (yes) | +9.63 | 34.53 ± 0.77 | 15.40 ± 0.60 |
| conv p=0.5 | 59.88 ± 0.27 | +3.70 pp (yes) | +7.30 | 20.08 ± 1.15 | 8.50 ± 0.61 |

## DropBlock block size

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| block 7 (paper) | 53.75 ± 0.73 | - | +1.18 | 46.25 ± 0.73 | 34.83 ± 0.74 |
| block 1 | 54.05 ± 1.04 | +0.29 pp (no) | +1.47 | 45.95 ± 1.04 | 36.43 ± 1.08 |
| block 3 | 55.45 ± 0.52 | +1.69 pp (yes) | +2.87 | 44.55 ± 0.52 | 33.57 ± 0.33 |
| block 5 | 54.16 ± 1.08 | +0.41 pp (no) | +1.58 | 45.84 ± 1.08 | 34.80 ± 0.88 |

## DropBlock placement

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| blocks 2-3 (paper) | 53.75 ± 0.73 | - | +1.18 | 46.25 ± 0.73 | 34.83 ± 0.74 |
| block 3 only | 53.47 ± 0.40 | -0.28 pp (no) | +0.89 | 46.53 ± 0.40 | 38.97 ± 0.50 |
| blocks 1-3 | 53.63 ± 1.23 | -0.12 pp (no) | +1.05 | 46.36 ± 1.23 | 31.91 ± 0.89 |

## DropBlock rate schedule

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| linear (paper) | 53.75 ± 0.73 | - | +1.18 | 46.25 ± 0.73 | 34.83 ± 0.74 |
| constant | 58.45 ± 0.44 | +4.69 pp (yes) | +5.87 | 41.55 ± 0.44 | 30.90 ± 0.26 |

Mean ± sample std over seeds, last-epoch models, validation set (the test set is not used). Conclusive when |Δ| > max(std of the variant, std of the selected configuration). The selected configurations were chosen on this validation set with seed 0, so their score is slightly optimistic.
