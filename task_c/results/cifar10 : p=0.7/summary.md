| Method | Seeds | Test acc (%) | Δ vs baseline (conclusive?) | Gen. gap (pp) | ECE (%) | Macro AUC | MC dropout acc (%) | MC dropout ECE (%) |
|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 50.71 ± 0.54 | - | 49.29 ± 0.54 | 42.42 ± 0.54 | 0.865 ± 0.004 | - | - |
| dropout | 3 | 55.55 ± 0.65 | +4.84 pp (yes) | 44.45 ± 0.65 | 36.93 ± 0.69 | 0.887 ± 0.002 | 55.57 ± 0.61 | 27.12 ± 0.91 |
| dropblock | 3 | 56.37 ± 0.69 | +5.66 pp (yes) | 43.63 ± 0.69 | 35.18 ± 0.70 | 0.893 ± 0.002 | 56.96 ± 0.54 | 23.34 ± 0.32 |

Reduced CIFAR-10, last-epoch models. Mean ± sample std over seeds. A difference is marked conclusive when |Δ| > max(std of the method, std of the baseline).
