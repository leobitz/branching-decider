"""Inference for trained JEV-Bench decision models (hierarchical or BEV).

Two surfaces over one checkpoint, each with a single-item and a batched form:
  * raw:       choice / noul / score          and  choice_batch / noul_batch / score_batch
  * JEV-style: decide(state, questions)       and  decide_batch(requests)

Noul and score questions are answered by averaging over the label surfaces seen in training
(yes/no, true/false, correct/incorrect, ... in both orders), which removes surface-form bias.
Batched forms flatten every variant of every item into one stream of model batches.
"""

from __future__ import annotations

import json
import math
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from jevbench.data.collators import build_collators
from jevbench.data.constants import NOUL_SURFACE_PAIRS, SCORE_SURFACE_STYLES
from jevbench.data.processing import (
    _criteria_as_key_desc, _criterion_description, _format_key_description, _format_score_labels,
    _make_noul_query, _parse_jsonish, humanize_state,
)
from jevbench.models import build_model, forward_batch
from jevbench.training.metrics import move_tensor_batch
from jevbench.utils.runtime import resolve_model_dtype

QUESTION_TYPES = ("choice", "score", "noul")


def _state_text(state):
    if state is None:
        text = ""
    else:
        text = state.strip() if isinstance(state, str) else humanize_state(state)
    return text or "No additional state was provided."


def _keys_and_descs(options, what):
    """Accept a list of labels or a {key: description} mapping; return parallel (keys, descs)."""
    options = _parse_jsonish(options)
    if isinstance(options, dict):
        keys, descs = _criteria_as_key_desc(options)
    elif isinstance(options, (list, tuple)):
        keys, descs = [str(o) for o in options], [""] * len(options)
    else:
        keys = []
    if len(keys) < 2:
        raise ValueError(f"{what} needs at least two options")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{what} option keys must be unique: {keys}")
    return keys, descs


def _distribution(probs):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("model returned a non-finite or empty distribution")
    return probs / total


def _example(context, query, labels, primitive):
    return {
        "context": context, "query": query, "labels": labels,
        "target_probs": [1.0 / len(labels)] * len(labels), "primitive": primitive,
    }


def _plan_choice(context, query, choices, order_tta=True):
    """Return (examples, finish); finish maps the examples' probabilities to the answer."""
    keys, descs = _keys_and_descs(choices, "choice")
    labels = [_format_key_description(k, d) for k, d in zip(keys, descs)]
    context = _state_text(context)

    perms = [list(range(len(keys)))]
    if order_tta:
        perms.append(perms[0][::-1])
    examples = [_example(context, query, [labels[i] for i in perm], "choice") for perm in perms]

    def finish(plist):
        probs = np.zeros(len(keys))
        for perm, p in zip(perms, plist):
            probs[perm] += p
        probs = _distribution(probs)
        best = int(probs.argmax())
        return {
            "type": "choice", "choice": keys[best],
            "probabilities": dict(zip(keys, probs.tolist())), "confidence": float(probs[best]),
        }

    return examples, finish


def _plan_noul(context, query, criteria=None):
    true_desc = false_desc = ""
    criteria = _parse_jsonish(criteria)
    for k, v in (criteria.items() if isinstance(criteria, dict) else ()):
        if str(k).strip().lower() in {"true", "1", "yes"}:
            true_desc = _criterion_description(v)
        elif str(k).strip().lower() in {"false", "0", "no"}:
            false_desc = _criterion_description(v)

    context, query = _state_text(context), _make_noul_query(query)
    examples, pos_index = [], []
    for pos, neg in NOUL_SURFACE_PAIRS:
        pos_label = _format_key_description(pos, true_desc)
        neg_label = _format_key_description(neg, false_desc)
        for labels, idx in (([pos_label, neg_label], 0), ([neg_label, pos_label], 1)):
            examples.append(_example(context, query, labels, "noul"))
            pos_index.append(idx)

    def finish(plist):
        return {"type": "noul", "noul": float(np.mean([p[i] for p, i in zip(plist, pos_index)]))}

    return examples, finish


def _plan_score(context, query, levels):
    keys, descs = _keys_and_descs(levels, "score")
    try:
        values = [float(k) for k in keys]
    except ValueError:
        values = [float(i) for i in range(len(keys))]
    context = _state_text(context)

    examples, seen = [], set()
    for style in SCORE_SURFACE_STYLES:
        labels, _ = _format_score_labels(keys, descs, style)
        if tuple(labels) not in seen:
            seen.add(tuple(labels))
            examples.append(_example(context, query, labels, "score"))

    def finish(plist):
        probs = _distribution(np.mean(plist, axis=0))
        return {
            "type": "score", "score": float(np.dot(probs, values)),
            "legend": dict(zip(keys, descs)),
            "probabilities": dict(zip(keys, probs.tolist())), "confidence": float(probs.max()),
        }

    return examples, finish


def _numbers(answer):
    for v in answer.values():
        if isinstance(v, float):
            yield v
        elif isinstance(v, dict):
            yield from (x for x in v.values() if isinstance(x, float))


def _plan_question(context, qid, question):
    """Validate one JEV question and plan it against an already-rendered state."""
    if not isinstance(question, dict):
        raise ValueError(f"question {qid!r} must be an object")
    kind = str(question.get("type", "")).lower()
    if kind not in QUESTION_TYPES:
        raise ValueError(f"question {qid!r}: type must be one of {QUESTION_TYPES}, got {kind!r}")
    instructions = str(question.get("instructions") or "").strip()
    if not instructions:
        raise ValueError(f"question {qid!r}: instructions are required")

    planner = {"choice": _plan_choice, "score": _plan_score, "noul": _plan_noul}[kind]
    examples, finish = planner(context, instructions, question.get("criteria"))

    def checked_finish(plist):
        answer = finish(plist)
        if not all(math.isfinite(v) for v in _numbers(answer)):
            raise ValueError(f"question {qid!r}: non-finite value in answer")
        return answer

    return examples, checked_finish


def _plan_request(state, questions):
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty mapping of question id -> question")
    context = _state_text(state)
    plans = {qid: _plan_question(context, qid, q) for qid, q in questions.items()}
    examples = [e for ex, _ in plans.values() for e in ex]

    def finish(plist):
        answers, i = {}, 0
        for qid, (ex, fin) in plans.items():
            answers[qid] = fin(plist[i:i + len(ex)])
            i += len(ex)
        return {"answers": answers}

    return examples, finish


class JevDecider:
    def __init__(self, model, tokenizer, args, device, batch_size=32):
        self.model = model.eval()
        self.args = args
        self.device = device
        self.batch_size = batch_size
        _, self.collator = build_collators(args, tokenizer)
        self.use_amp = device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    @classmethod
    def from_checkpoint(cls, checkpoint, device=None, model_dtype=None, batch_size=32):
        """`checkpoint`: a best_val dir, an output dir containing one, or a Hugging Face repo id."""
        path = Path(checkpoint)
        if not path.exists() and not path.is_absolute() and str(checkpoint).count("/") == 1:
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(str(checkpoint)))
        if not (path / "model.pt").exists() and (path / "best_val" / "model.pt").exists():
            path = path / "best_val"
        if not (path / "model.pt").exists():
            raise FileNotFoundError(f"no model.pt under {path}")

        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        args = Namespace(**config)
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = resolve_model_dtype(device, model_dtype or config.get("model_dtype", "auto"))

        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = build_model(args, dtype)
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        return cls(model.to(device), tokenizer, args, device, batch_size)

    @torch.no_grad()
    def _probabilities(self, examples):
        """Run examples through the model; return one probability array per example (label order)."""
        out = []
        for i in range(0, len(examples), self.batch_size):
            chunk = examples[i:i + self.batch_size]
            batch = move_tensor_batch(self.collator(chunk), self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                result = forward_batch(self.model, batch, self.args.architecture)
            probs = result["probabilities"].float().cpu().numpy()
            out.extend(_distribution(p[:len(ex["labels"])]) for p, ex in zip(probs, chunk))
        return out

    def _run(self, plans):
        """Evaluate every plan's examples in one pass, then split results back per plan."""
        probs = self._probabilities([e for ex, _ in plans for e in ex])
        results, i = [], 0
        for ex, finish in plans:
            results.append(finish(probs[i:i + len(ex)]))
            i += len(ex)
        return results

    # Batched: items are dicts. choice: {context, query, choices}; noul: {context, query, criteria?};
    # score: {context, query, levels}.
    def choice_batch(self, items, order_tta=True):
        return self._run([_plan_choice(it.get("context"), it["query"], it["choices"], order_tta) for it in items])

    def noul_batch(self, items):
        return self._run([_plan_noul(it.get("context"), it["query"], it.get("criteria")) for it in items])

    def score_batch(self, items):
        return self._run([_plan_score(it.get("context"), it["query"], it["levels"]) for it in items])

    def decide_batch(self, requests):
        """requests: [{"state": ..., "questions": {...}}, ...] -> [{"answers": {...}}, ...]."""
        return self._run([_plan_request(r.get("state"), r.get("questions")) for r in requests])

    # Single-item forms.
    def choice(self, context, query, choices, order_tta=True):
        """`choices`: list of labels or {key: description}. Averages original and reversed order."""
        return self.choice_batch([{"context": context, "query": query, "choices": choices}], order_tta)[0]

    def noul(self, context, query, criteria=None):
        """P(yes) averaged over every yes/no surface pair in both orders.

        `criteria` may hold true/false descriptions, e.g. {"true": "...", "false": "..."}.
        """
        return self.noul_batch([{"context": context, "query": query, "criteria": criteria}])[0]

    def score(self, context, query, levels):
        """Probability-weighted score over ordered `levels` ({key: description} or list of labels).

        The score is the expected level value: numeric keys are used as values, otherwise 0-based indices.
        """
        return self.score_batch([{"context": context, "query": query, "levels": levels}])[0]

    def decide(self, state, questions):
        """JEV-style API: evaluate every question against one shared state.

        questions = {id: {"type": "choice"|"score"|"noul", "instructions": str, "criteria": {...}}}
        """
        return self.decide_batch([{"state": state, "questions": questions}])[0]
