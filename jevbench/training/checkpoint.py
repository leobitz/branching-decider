"""Checkpoint management."""

import json
from pathlib import Path

import torch

class BestCheckpointManager:
    """Keeps a single best checkpoint for one monitored metric, overwriting on improvement."""

    def __init__(self, output_dir: Path, name: str, mode: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.name = name  # e.g. "best_val" or "best_ood"
        self.mode = mode  # "max" or "min"
        self.best_score = None
        self.best_loss = None
        self.best_step = None
        self.path = self.output_dir / name

    def _is_better(self, score, loss, step):
        if self.best_score is None:
            return True
        if self.mode == "max":
            if score != self.best_score:
                return score > self.best_score
        else:
            if score != self.best_score:
                return score < self.best_score
        if loss != self.best_loss:
            return loss < self.best_loss
        return step > self.best_step

    def maybe_save(self, score, loss, step, model, tokenizer, config):
        if not self._is_better(score, loss, step):
            return None

        if self.path.is_symlink():
            self.path.unlink()
        elif self.path.exists():
            _remove_dir(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.path / "model.pt")
        tokenizer.save_pretrained(self.path)
        meta = dict(config)
        meta["checkpoint"] = {"name": self.name, "score": score, "loss": loss, "step": step}
        (self.path / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.best_score = score
        self.best_loss = loss
        self.best_step = step
        return self.path


def _remove_dir(path: Path):
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            child.rmdir()
    path.rmdir()
