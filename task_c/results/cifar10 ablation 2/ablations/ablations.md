# Ablations (validation set, reduced CIFAR-10)

Baseline: validation accuracy 50.64 ± 0.55 %, ECE 42.57 ± 0.66 %.

## Dropout rate beyond the grid (dense)

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| p=0.7 (selected) | 55.46 ± 0.60 | - | +4.83 | 44.54 ± 0.60 | 37.12 ± 0.48 |
| p=0.8 | 55.82 ± 1.01 | +0.35 pp (no) | +5.18 | 44.18 ± 1.01 | 36.44 ± 1.00 |
| p=0.9 | 54.12 ± 1.20 | -1.34 pp (yes) | +3.49 | 45.85 ± 1.20 | 35.69 ± 0.68 |

## Dropout position at a fair rate

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| dense p=0.7 (selected) | 55.46 ± 0.60 | - | +4.83 | 44.54 ± 0.60 | 37.12 ± 0.48 |
| conv p=0.1 | 55.19 ± 1.09 | -0.27 pp (no) | +4.55 | 44.81 ± 1.09 | 37.38 ± 0.84 |
| conv p=0.2 | 56.43 ± 1.88 | +0.97 pp (no) | +5.80 | 43.57 ± 1.88 | 34.55 ± 1.40 |
| conv p=0.3 | 57.96 ± 3.45 | +2.50 pp (no) | +7.32 | 42.04 ± 3.45 | 31.51 ± 2.65 |
| conv p=0.4 | 59.95 ± 1.01 | +4.49 pp (yes) | +9.32 | 40.05 ± 1.01 | 26.92 ± 0.65 |
| conv p=0.5 | 59.49 ± 2.23 | +4.03 pp (yes) | +8.85 | 40.22 ± 2.10 | 21.92 ± 1.36 |

## DropBlock block size (p=0.2)

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| block 3 (selected) | 56.88 ± 0.48 | - | +6.24 | 43.12 ± 0.48 | 34.56 ± 0.52 |
| block 1 | 53.97 ± 1.42 | -2.90 pp (yes) | +3.33 | 46.03 ± 1.42 | 36.81 ± 1.19 |
| block 5 | 54.45 ± 0.43 | -2.42 pp (yes) | +3.81 | 45.55 ± 0.43 | 37.02 ± 0.43 |
| block 7 | 55.37 ± 0.81 | -1.50 pp (yes) | +4.73 | 44.63 ± 0.81 | 36.02 ± 0.68 |

## DropBlock placement (p=0.2, block 3)

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| blocks 2-3 (selected) | 56.88 ± 0.48 | - | +6.24 | 43.12 ± 0.48 | 34.56 ± 0.52 |
| block 3 only | 54.00 ± 1.17 | -2.87 pp (yes) | +3.37 | 46.00 ± 1.17 | 38.04 ± 1.16 |
| blocks 1-3 | 55.83 ± 1.00 | -1.05 pp (yes) | +5.19 | 44.18 ± 1.00 | 34.43 ± 0.78 |

## Dropout + DropBlock

| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |
|---|---|---|---|---|---|
| dropout only (selected) | 55.46 ± 0.60 | - | +4.83 | 44.54 ± 0.60 | 37.12 ± 0.48 |
| + DropBlock p=0.2 | 59.20 ± 1.54 | +3.73 pp (yes) | +8.56 | 40.80 ± 1.54 | 30.75 ± 1.00 |

Mean ± sample std over seeds, last-epoch models, validation set (the test set is not used). Conclusive when |Δ| > max(std of the variant, std of the selected configuration). The selected configurations were chosen on this validation set with seed 0, so their score is slightly optimistic.
