"""Model parameter reporting."""

from __future__ import annotations


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _fmt(n):
    if n >= 1_000_000_000:
        return f"{n:,} ({n / 1_000_000_000:.3f}B)"
    if n >= 1_000_000:
        return f"{n:,} ({n / 1_000_000:.3f}M)"
    if n >= 1_000:
        return f"{n:,} ({n / 1_000:.1f}K)"
    return f"{n:,}"


def print_parameter_summary(model, architecture):
    total, trainable = count_parameters(model)
    frozen = total - trainable
    pct = 100.0 * trainable / max(total, 1)

    print("\nParameter summary")
    print("-" * 78)
    print(f"architecture : {architecture}")
    print(f"total        : {_fmt(total)}")
    print(f"trainable    : {_fmt(trainable)} ({pct:.2f}%)")
    print(f"frozen       : {_fmt(frozen)}")
    print("\nTop-level components:")
    for name, child in model.named_children():
        child_total, child_trainable = count_parameters(child)
        child_pct = 100.0 * child_trainable / max(child_total, 1)
        print(
            f"  {name:<24} total={_fmt(child_total):>20}  "
            f"trainable={_fmt(child_trainable):>20}  ({child_pct:6.2f}%)"
        )
    print("-" * 78 + "\n")
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": pct,
    }
