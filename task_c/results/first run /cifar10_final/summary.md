| Method | Seeds | Test acc (%) | Δ vs baseline (conclusive?) | Weighted F1 (%) | Δ F1 (conclusive?) | Gen. gap (pp) | ECE (%) | Macro AUC | MC dropout acc (%) | MC dropout ECE (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 50.71 ± 0.54 | - | 50.45 ± 0.73 | - | 49.29 ± 0.54 | 42.42 ± 0.54 | 0.865 ± 0.004 | - | - |
| dropout | 3 | 55.55 ± 0.65 | +4.84 pp (yes) | 55.43 ± 0.69 | +4.98 pp (yes) | 44.45 ± 0.65 | 36.93 ± 0.69 | 0.887 ± 0.002 | 55.57 ± 0.61 | 27.12 ± 0.91 |
| dropblock | 3 | 53.10 ± 0.62 | +2.39 pp (yes) | 52.59 ± 0.64 | +2.14 pp (yes) | 46.90 ± 0.62 | 37.12 ± 0.55 | 0.877 ± 0.002 | 54.02 ± 0.39 | 18.17 ± 0.23 |
| dropout+dropblock | 3 | 57.96 ± 0.41 | +7.25 pp (yes) | 57.67 ± 0.45 | +7.22 pp (yes) | 42.04 ± 0.41 | 29.36 ± 0.49 | 0.901 ± 0.000 | 57.98 ± 0.38 | 9.80 ± 0.67 |

Reduced CIFAR-10, last-epoch models. Mean ± sample std over seeds. A difference is marked conclusive when |Δ| > max(std of the method, std of the baseline). Accuracy is the brief's common result; weighted F1 is the metric reported by Opperman et al. (Table IV), for direct comparison.
