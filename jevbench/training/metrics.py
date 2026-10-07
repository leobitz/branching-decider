"""Shared loss, metrics, and evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from jevbench.models import forward_batch


def move_tensor_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def distribution_cross_entropy(logits, target_probs, candidate_mask):
    z = logits.masked_fill(~candidate_mask, -1e9)
    log_p = F.log_softmax(z, dim=-1)
    return -(target_probs * log_p).sum(dim=-1).mean()


def argmax_accuracy(logits, target_probs):
    return (logits.argmax(-1) == target_probs.argmax(-1)).float().mean().item()


def raise_nonfinite_forward(out, loss, step):
    if torch.isfinite(loss).item():
        return
    bad = []
    for name, x in out.items():
        if torch.is_tensor(x) and not torch.isfinite(x).all().item():
            bad.append(name)
    raise FloatingPointError(
        f"Non-finite training loss at micro-step {step}. "
        f"Non-finite forward tensors: {bad or ['none detected']}."
    )


@torch.no_grad()
def evaluate(model, loader, device, use_amp, amp_dtype, architecture, max_batches=None):
    model.eval()
    losses, correct, total = [], 0, 0
    nll_sum = brier_sum = 0.0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        b = move_tensor_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = forward_batch(model, b, architecture)
            loss = distribution_cross_entropy(out["logits"], b["target_probs"], b["candidate_mask"])
        losses.append(float(loss.item()))
        probs = out["probabilities"].float()
        target_probs = b["target_probs"].float()
        pred = out["logits"].argmax(-1)
        gold = target_probs.argmax(-1)
        correct += int((pred == gold).sum().item())
        total += len(pred)
        safe = probs.clamp_min(1e-12)
        nll_sum += float((-(target_probs * safe.log()).sum(-1)).sum().item())
        diff = (probs - target_probs) * b["candidate_mask"]
        brier_sum += float((diff * diff).sum(-1).sum().item())
    n = max(total, 1)
    model.train()
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": correct / n,
        "nll": nll_sum / n,
        "brier": brier_sum / n,
    }
