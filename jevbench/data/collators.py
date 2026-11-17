"""Architecture-specific minibatch collation."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from .constants import TASK_TYPES
from .bev_formatting import bev_encode_example


def hierarchical_prefix(context, query):
    context, query = str(context or ""), str(query or "")
    if context.strip():
        return f"Context: {context}\nQuery: {query}\nCandidate:"
    return f"Query: {query}\nCandidate:"


def hierarchical_token_length(tokenizer, context, query, labels, max_label_length):
    """Untruncated tokens the hierarchical collator needs: shared prefix + longest candidate branch."""
    n_prefix = len(tokenizer(hierarchical_prefix(context, query), add_special_tokens=True)["input_ids"])
    n_branch = max(
        len(tokenizer(" " + str(label), add_special_tokens=False, truncation=True,
                      max_length=max_label_length - 1)["input_ids"]) + 1
        for label in labels
    )
    return n_prefix + n_branch


@dataclass
class BranchingVariableChoiceCollator:
    """Build one shared prefix per example and one flattened row per candidate branch.

    There are two independent masks:
      * branch_attention_mask [M, L] handles variable candidate-token lengths;
      * candidate_mask [B, N] handles variable candidate counts.

    M is the number of real candidates across the minibatch. No padded candidate
    rows are sent through Qwen.
    """

    tokenizer: object
    max_length: int = 512
    max_label_length: int = 64
    training: bool = False
    randomize_order: bool = True
    seed: int = 42

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        if self.max_label_length < 2:
            raise ValueError("max_label_length must be >= 2 (at least one label token + EOS)")
        if self.max_length <= self.max_label_length:
            raise ValueError("max_length must be larger than max_label_length")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Qwen tokenizer must define eos_token_id")
        if self.tokenizer.pad_token_id is None:
            # Qwen tokenizers normally already define this, but make the
            # collator robust to a locally modified tokenizer.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def _transform_choices(self, labels, target_probs):
        labels = list(labels)
        probs = np.asarray(target_probs, dtype=np.float32)
        if len(labels) != len(probs):
            raise ValueError("labels and target_probs must have the same length")

        perm = list(range(len(labels)))
        if self.training and self.randomize_order:
            self.rng.shuffle(perm)

        new_labels = [labels[i] for i in perm]
        new_probs = probs[perm]
        new_probs = new_probs / max(float(new_probs.sum()), 1e-12)
        new_target = int(np.argmax(new_probs))
        return new_labels, new_target, new_probs, perm

    def __call__(self, batch):
        transformed = []
        max_n = 0
        for ex in batch:
            labels, target, probs, perm = self._transform_choices(ex["labels"], ex["target_probs"])
            transformed.append((ex, labels, target, probs, perm))
            max_n = max(max_n, len(labels))

        prefixes = []
        branch_token_lists = []
        branch_to_example = []
        choice_index = []
        targets = []
        target_probs = torch.zeros(len(batch), max_n, dtype=torch.float32)
        candidate_mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
        choice_texts = []

        for b, (ex, labels, target, probs, perm) in enumerate(transformed):
            targets.append(target)
            choice_texts.append(labels)
            target_probs[b, : len(labels)] = torch.tensor(probs, dtype=torch.float32)

            prefixes.append(hierarchical_prefix(ex.get("context", ""), ex.get("query", "")))

            for j, label in enumerate(labels):
                # Candidate is a suffix of the shared prefix. A common terminal
                # EOS state is gathered as the fixed-width candidate embedding.
                ids = self.tokenizer(
                    " " + str(label),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_label_length - 1,
                )["input_ids"]
                ids = list(ids) + [self.tokenizer.eos_token_id]
                branch_token_lists.append(ids)
                branch_to_example.append(b)
                choice_index.append(j)
                candidate_mask[b, j] = True

        # Use the actual longest branch in this minibatch when reserving the
        # shared-context budget. This avoids unnecessarily truncating context
        # when all current labels are short while still enforcing max_length.
        max_branch_len = max(len(ids) for ids in branch_token_lists)
        max_context_length = self.max_length - max_branch_len
        if max_context_length < 1:
            raise ValueError("Current candidate branches leave no room for context tokens")

        context_tok = self.tokenizer(
            prefixes,
            padding=True,
            truncation=True,
            max_length=max_context_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        pad_id = int(self.tokenizer.pad_token_id)
        branch_input_ids = torch.full(
            (len(branch_token_lists), max_branch_len),
            pad_id,
            dtype=torch.long,
        )
        branch_attention_mask = torch.zeros(
            (len(branch_token_lists), max_branch_len),
            dtype=torch.bool,
        )
        for i, ids in enumerate(branch_token_lists):
            n = len(ids)
            branch_input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            branch_attention_mask[i, :n] = True

        return {
            "context_input_ids": context_tok["input_ids"],
            "context_attention_mask": context_tok["attention_mask"].to(torch.bool),
            "branch_input_ids": branch_input_ids,
            "branch_attention_mask": branch_attention_mask,
            "branch_to_example": torch.tensor(branch_to_example, dtype=torch.long),
            "choice_index": torch.tensor(choice_index, dtype=torch.long),
            "candidate_mask": candidate_mask,
            "targets": torch.tensor(targets, dtype=torch.long),
            "target_probs": target_probs,
            "choice_texts": choice_texts,
            "raw_examples": [x[0] for x in transformed],
        }


@dataclass
class BEVJevCollator:
    tokenizer: object
    max_state_tokens: int = 512
    max_choice_tokens: int = 96
    max_packed_tokens: int = 32768
    training: bool = False
    randomize_order: bool = True
    seed: int = 42

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def _transform_choices(self, labels, target_probs):
        labels = list(labels)
        probs = np.asarray(target_probs, dtype=np.float32)
        if len(labels) != len(probs):
            raise ValueError("labels and target_probs must have the same length")

        perm = list(range(len(labels)))
        if self.training and self.randomize_order:
            self.rng.shuffle(perm)
        labels = [labels[i] for i in perm]
        probs = probs[perm]
        probs = probs / max(float(probs.sum()), 1e-12)
        return labels, int(np.argmax(probs)), probs, perm

    def __call__(self, batch):
        encoded = []
        for ex in batch:
            choices, target, probs, perm = self._transform_choices(ex["labels"], ex["target_probs"])
            primitive = str(ex.get("primitive", "choice")).lower()
            if primitive not in TASK_TYPES:
                raise ValueError(f"Unsupported BEV task type {primitive!r}")

            item = bev_encode_example(
                self.tokenizer,
                state=str(ex.get("context", "")),
                instructions=str(ex.get("query", "")),
                choices=choices,
                task_type=primitive,
                max_state_tokens=self.max_state_tokens,
                max_choice_tokens=self.max_choice_tokens,
            )
            if len(item["input_ids"]) > self.max_packed_tokens:
                raise ValueError(
                    f"packed BEV example has {len(item['input_ids']):,} tokens > "
                    f"--max-packed-tokens={self.max_packed_tokens:,}; "
                    f"source_config={ex.get('source_config')} choices={len(choices)}"
                )
            encoded.append((ex, choices, target, probs, primitive, item))

        B = len(encoded)
        L = max(len(x[-1]["input_ids"]) for x in encoded)
        C = max(len(x[-1]["choice_read_idx"]) for x in encoded)

        input_ids = torch.full((B, L), int(self.tokenizer.pad_token_id), dtype=torch.long)
        position_ids = torch.zeros((B, L), dtype=torch.long)
        # Exact BEV padding convention: padded query rows self-attend to avoid all -inf rows.
        attention_mask = torch.eye(L).repeat(B, 1, 1)
        choice_read_idx = torch.zeros((B, C), dtype=torch.long)
        choice_mask = torch.zeros((B, C), dtype=torch.bool)
        answer_end_idx = torch.zeros(B, dtype=torch.long)
        task_type = torch.zeros(B, dtype=torch.long)
        target_probs = torch.zeros((B, C), dtype=torch.float32)
        targets = torch.zeros(B, dtype=torch.long)
        choice_texts = []

        for b, (ex, choices, target, probs, primitive, item) in enumerate(encoded):
            n = len(item["input_ids"])
            c = len(item["choice_read_idx"])
            input_ids[b, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            position_ids[b, :n] = torch.tensor(item["position_ids"], dtype=torch.long)
            attention_mask[b, :n, :n] = item["attention_mask"]
            choice_read_idx[b, :c] = torch.tensor(item["choice_read_idx"], dtype=torch.long)
            choice_mask[b, :c] = True
            answer_end_idx[b] = int(item["answer_end_idx"])
            task_type[b] = TASK_TYPES[primitive]
            target_probs[b, :c] = torch.tensor(probs, dtype=torch.float32)
            targets[b] = int(target)
            choice_texts.append(choices)

        attention_mask = torch.where(attention_mask == 0, -float("inf"), 0.0).unsqueeze(1)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "choice_read_idx": choice_read_idx,
            "choice_mask": choice_mask,
            "answer_end_idx": answer_end_idx,
            "task_type": task_type,
            "candidate_mask": choice_mask,  # compatibility alias for shared CE/metrics code
            "targets": targets,
            "target_probs": target_probs,
            "choice_texts": choice_texts,
            "raw_examples": [x[0] for x in encoded],
        }


def build_collators(args, tokenizer):
    """Return (train_collator, eval_collator) for the selected architecture."""
    if args.architecture in ("hierarchical", "adaptive"):
        common = dict(
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_label_length=args.max_label_length,
        )
        return (
            BranchingVariableChoiceCollator(
                **common, training=True, randomize_order=args.randomize_candidate_order, seed=args.seed + 1
            ),
            BranchingVariableChoiceCollator(
                **common, training=False, randomize_order=False, seed=args.seed + 2
            ),
        )
    if args.architecture == "bev":
        common = dict(
            tokenizer=tokenizer,
            max_state_tokens=args.max_state_tokens,
            max_choice_tokens=args.max_choice_tokens,
            max_packed_tokens=args.max_packed_tokens,
        )
        return (
            BEVJevCollator(
                **common, training=True, randomize_order=args.randomize_candidate_order, seed=args.seed + 1
            ),
            BEVJevCollator(
                **common, training=False, randomize_order=False, seed=args.seed + 2
            ),
        )
    raise ValueError(f"Unknown architecture: {args.architecture}")
