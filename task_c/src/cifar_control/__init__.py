"""Task C - reduced CIFAR-10 control for the G2 dropout study.

runs the PathMNIST protocol unchanged on non-medical images (baseline, dropout, DropBlock),
so that differences with Opperman et al. (2026) can be attributed to the domain or to the protocol.

Modules
-------
config     typed configs (YAML + inheritance + overrides), run ids
data       reduced CIFAR-10: balanced 4,500-image subset, validation, test
layers     DropBlock
models     the protocol CNN with optional dropout / DropBlock
train      training loop (SGD + momentum, cosine LR, last-epoch model)
runs       run directories, caching, sweeps and selection on validation
metrics    accuracy, NLL, ECE, macro AUC, reliability bins
evaluate   test evaluation (once), MC dropout
reporting  summary table (mean +/- std over seeds, delta vs baseline) and figures
pipeline   the stages run by scripts/run.py
"""

__version__ = "0.1.0"
