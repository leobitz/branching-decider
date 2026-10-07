"""Balanced sampling utilities."""

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler

class BalancedGroupSampler(Sampler):
    """Draw an (almost) exactly equal number of examples from every group each epoch."""

    def __init__(self, rows, group_field="task_family", num_samples=None, seed=42):
        self.group_field = str(group_field)
        self.num_samples = int(num_samples) if num_samples is not None else len(rows)
        self.rng = np.random.default_rng(seed)
        self.groups = defaultdict(list)
        for idx, row in enumerate(rows):
            self.groups[str(row.get(self.group_field, "unknown"))].append(idx)

        self.group_names = sorted(self.groups)
        if not self.group_names:
            raise ValueError("Cannot build a balanced sampler for an empty training dataset.")
        if self.num_samples < len(self.group_names):
            raise ValueError(
                f"train_samples_per_epoch={self.num_samples} is smaller than the number "
                f"of {self.group_field} groups ({len(self.group_names)})."
            )

    def __iter__(self):
        base, remainder = divmod(self.num_samples, len(self.group_names))
        extra_groups = set()
        if remainder:
            chosen = self.rng.choice(len(self.group_names), size=remainder, replace=False)
            extra_groups = {self.group_names[int(i)] for i in chosen}

        selected = []
        for group in self.group_names:
            quota = base + int(group in extra_groups)
            indices = np.asarray(self.groups[group], dtype=np.int64)
            replace = quota > len(indices)
            picks = self.rng.choice(indices, size=quota, replace=replace)
            selected.extend(int(i) for i in picks)

        self.rng.shuffle(selected)
        return iter(selected)

    def __len__(self):
        return self.num_samples
