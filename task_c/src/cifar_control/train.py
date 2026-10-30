"""Training loop of the common protocol.

Cross-entropy, SGD with momentum, cosine learning-rate decay stepped once per epoch, fixed number
of epochs, no early stopping: the model kept is the one from the last epoch. The whole subset
lives on the device and is shuffled with a seeded generator, so a run is reproducible from its seed.
"""

from __future__ import annotations

import logging
import math
import time

import torch
import torch.nn.functional as F
from torch import nn

from .config import TrainConfig
from .data import Split

log = logging.getLogger(__name__)


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "constant":
        return None
    raise ValueError(f"unknown scheduler '{cfg.scheduler}'")


@torch.no_grad()
def loss_and_accuracy(model: nn.Module, split: Split, batch_size: int = 1024) -> tuple[float, float]:
    """Mean cross-entropy and accuracy in eval mode."""
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=split.x.device)
    correct = torch.zeros((), device=split.x.device)
    for i in range(0, len(split), batch_size):
        logits = model(split.x[i:i + batch_size])
        y = split.y[i:i + batch_size]
        total_loss += F.cross_entropy(logits, y, reduction="sum")
        correct += (logits.argmax(1) == y).sum()
    model.train(was_training)
    return total_loss.item() / len(split), correct.item() / len(split)


def train_model(model: nn.Module, train: Split, val: Split, cfg: TrainConfig, seed: int,
                log_every: int = 10) -> dict:
    """Train in place. Returns the per-epoch history.

    ``train_loss`` / ``train_acc`` are running averages over the epoch with regularisation active;
    ``val_loss`` / ``val_acc`` are measured in eval mode at the end of each epoch.
    """
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    generator = torch.Generator().manual_seed(seed)
    device = train.x.device
    n = len(train)

    history: dict = {k: [] for k in ("epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc")}
    history["diverged"] = False
    scheduled = [m for m in model.modules() if hasattr(m, "set_progress")]  # scheduled DropBlock
    total_steps = cfg.epochs * math.ceil(n / cfg.batch_size)
    step = 0
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for i in range(0, n, cfg.batch_size):
            for module in scheduled:
                module.set_progress(step / total_steps)
            step += 1
            idx = order[i:i + cfg.batch_size]
            x, y = train.x[idx], train.y[idx]
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:  # caps a single update; same for every method (group protocol)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.detach() * len(idx)
            correct += (logits.detach().argmax(1) == y).sum()

        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        train_loss = total_loss.item() / n
        val_loss, val_acc = loss_and_accuracy(model, val)
        for key, value in (("epoch", epoch), ("lr", lr), ("train_loss", train_loss),
                           ("train_acc", correct.item() / n), ("val_loss", val_loss), ("val_acc", val_acc)):
            history[key].append(value)

        if not math.isfinite(train_loss):
            history["diverged"] = True
            log.warning("training diverged at epoch %d (loss=%s); stopping this run", epoch, train_loss)
            break
        if epoch == 1 or epoch % log_every == 0 or epoch == cfg.epochs:
            log.info("epoch %3d/%d  lr %.4f  train loss %.4f acc %.3f | val loss %.4f acc %.3f",
                     epoch, cfg.epochs, lr, train_loss, history["train_acc"][-1], val_loss, val_acc)

    for module in scheduled:  # leave every scheduled layer at its target rate (MC dropout uses it)
        module.set_progress(1.0)
    history["seconds"] = time.perf_counter() - start
    return history