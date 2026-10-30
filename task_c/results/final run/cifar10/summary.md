| Method | Seeds | Test acc (%) | Δ vs baseline (conclusive?) | Weighted F1 (%) | Δ F1 (conclusive?) | Gen. gap (pp) | ECE (%) | Macro AUC | MC dropout acc (%) | MC dropout ECE (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 52.22 ± 1.06 | - | 52.06 ± 1.00 | - | 47.78 ± 1.06 | 40.78 ± 0.94 | 0.878 ± 0.003 | - | - |
| dropout | 3 | 55.51 ± 0.41 | +3.30 pp (yes) | 55.30 ± 0.44 | +3.23 pp (yes) | 44.49 ± 0.41 | 37.29 ± 0.37 | 0.895 ± 0.002 | 55.49 ± 0.30 | 26.35 ± 0.17 |
| dropblock | 3 | 53.94 ± 0.73 | +1.73 pp (yes) | 53.41 ± 0.71 | +1.34 pp (yes) | 46.06 ± 0.73 | 34.47 ± 0.43 | 0.891 ± 0.002 | 54.49 ± 0.66 | 11.27 ± 0.17 |
| dropout+dropblock | 3 | 57.95 ± 0.28 | +5.74 pp (yes) | 57.50 ± 0.37 | +5.44 pp (yes) | 42.05 ± 0.28 | 27.59 ± 0.23 | 0.909 ± 0.001 | 58.30 ± 0.43 | 5.50 ± 0.31 |

Reduced CIFAR-10, last-epoch models. Mean ± sample std over seeds. A difference is marked conclusive when |Δ| > max(std of the method, std of the baseline). Accuracy is the brief's common result; weighted F1 is the metric reported by Opperman et al. (Table IV), for direct comparison.
