"""JEV-Bench row conversion into a common text/candidate representation."""

from __future__ import annotations

import ast
import hashlib
import json

import numpy as np

from .constants import (
    CONFIG_DOMAINS, DATASET_NAME_DEFAULT, NOUL_SURFACE_PAIRS, SCORE_SURFACE_STYLES,
)

def _stable_int(*parts, modulo=2**31 - 1):
    text = "||".join(str(x) for x in parts)
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % modulo


def _parse_jsonish(value):
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(text)
            except Exception:
                pass
        return value
    return value


def _humanize_key(key: str) -> str:
    key = str(key).strip().replace("_", " ")
    replacements = {
        " id": " ID",
        " url": " URL",
        " uri": " URI",
        " api": " API",
        " ms": " (milliseconds)",
        " s": " (seconds)",
    }
    low = " " + key.lower()
    for suffix, repl in replacements.items():
        if low.endswith(suffix):
            key = key[: -len(suffix.strip())] + repl
            break
    return key[:1].upper() + key[1:] if key else key


def _scalar_text(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def humanize_state(value, heading=None, level=0) -> str:
    """Deterministically render arbitrary JSON-like state as readable structured text."""
    value = _parse_jsonish(value)
    if isinstance(value, str):
        if heading:
            return f"{_humanize_key(heading)}:\n{value.strip()}"
        return value.strip()

    lines = []
    if heading is not None:
        lines.append(f"{_humanize_key(heading)}:")

    if isinstance(value, dict):
        for key, child in value.items():
            pretty = _humanize_key(key)
            if isinstance(child, dict):
                if lines and lines[-1] != "":
                    lines.append("")
                lines.append(f"{pretty}:")
                nested = humanize_state(child, heading=None, level=level + 1)
                lines.extend(nested.splitlines())
            elif isinstance(child, list):
                if lines and lines[-1] != "":
                    lines.append("")
                lines.append(f"{pretty}:")
                if not child:
                    lines.append("- none")
                else:
                    for item in child:
                        if isinstance(item, (dict, list)):
                            rendered = humanize_state(item, heading=None, level=level + 1).splitlines()
                            if rendered:
                                lines.append(f"- {rendered[0]}")
                                lines.extend(f"  {x}" for x in rendered[1:])
                        else:
                            lines.append(f"- {_scalar_text(item)}")
            else:
                lines.append(f"{pretty}: {_scalar_text(child)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                rendered = humanize_state(item, heading=None, level=level + 1).splitlines()
                if rendered:
                    lines.append(f"- {rendered[0]}")
                    lines.extend(f"  {x}" for x in rendered[1:])
            else:
                lines.append(f"- {_scalar_text(item)}")
    else:
        lines.append(_scalar_text(value))

    # Collapse excessive blank lines while keeping section readability.
    out = []
    for line in lines:
        if line == "" and (not out or out[-1] == ""):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _question_instruction(question) -> str:
    if isinstance(question, str):
        return question.strip()
    if not isinstance(question, dict):
        return str(question)
    for key in ("instructions", "instruction", "query", "question", "prompt"):
        v = question.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return "Choose the best answer from the provided candidates."


def _criterion_description(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("description", "text", "label", "name"):
            if value.get(key) is not None:
                return str(value[key]).strip()
        return humanize_state(value)
    return str(value).strip()


def _criteria_as_key_desc(criteria):
    """Return (source_keys, descriptions) while preserving source criterion order."""
    criteria = _parse_jsonish(criteria)
    if isinstance(criteria, dict):
        keys = [str(k) for k in criteria.keys()]
        descs = [_criterion_description(criteria[k]) for k in criteria.keys()]
        return keys, descs
    if isinstance(criteria, (list, tuple)):
        keys = [str(i) for i in range(len(criteria))]
        descs = [_criterion_description(x) for x in criteria]
        return keys, descs
    return [], []


def _format_key_description(key, desc):
    key = str(key).strip()
    desc = str(desc).strip()
    if not desc or desc.lower() == key.lower():
        return key
    return f"{key} - {desc}"


def _source_options_from_question(question):
    if not isinstance(question, dict):
        return [], []
    for name in ("criteria", "options", "choices", "labels"):
        if name in question and question[name] is not None:
            keys, descs = _criteria_as_key_desc(question[name])
            if keys:
                return keys, descs
    return [], []


def _format_choice_labels(keys, descs):
    return [_format_key_description(k, d) for k, d in zip(keys, descs)]


def _format_score_labels(keys, descs, style_name):
    k_total = len(keys)
    labels = []
    for i, (key, desc) in enumerate(zip(keys, descs)):
        desc = str(desc).strip()
        if style_name == "key_description":
            text = _format_key_description(key, desc)
        elif style_name == "score_key_description":
            text = f"score {key} - {desc}" if desc else f"score {key}"
        elif style_name == "level_key_description":
            text = f"level {key} - {desc}" if desc else f"level {key}"
        elif style_name == "rating_position_description":
            text = f"rating {i + 1} of {k_total} - {desc}" if desc else f"rating {i + 1} of {k_total}"
        elif style_name == "description_only":
            text = desc or str(key)
        elif style_name == "option_position_description":
            text = f"option {i + 1} of {k_total} - {desc}" if desc else f"option {i + 1} of {k_total}"
        else:
            raise ValueError(style_name)
        labels.append(text)
    # Description-only is unsafe if descriptions collide.
    if len(set(labels)) != len(labels):
        labels = [_format_key_description(k, d) for k, d in zip(keys, descs)]
        style_name = "key_description_fallback"
    return labels, style_name


def _normalize_noul_gold(raw_label):
    s = str(raw_label).strip().lower()
    if s in {"1", "true", "yes", "y", "positive"}:
        return True
    if s in {"0", "false", "no", "n", "negative"}:
        return False
    try:
        return bool(int(float(s)))
    except Exception as exc:
        raise ValueError(f"Unsupported Noul label: {raw_label!r}") from exc


def _make_noul_query(instruction: str) -> str:
    instruction = str(instruction).strip()
    if instruction.endswith("?"):
        return instruction
    return f"Determine whether the following statement is correct:\n{instruction}"


def _soft_probs_from_source(soft_label, primitive, source_keys):
    """Map optional human label distribution onto source option order."""
    soft = _parse_jsonish(soft_label)
    if soft in (None, "", {}):
        return None
    if primitive == "noul":
        # Common forms: {false: p0, true: p1}, {0: p0, 1: p1}, or a scalar P(true).
        if isinstance(soft, dict):
            false_v = None
            true_v = None
            for k, v in soft.items():
                kl = str(k).strip().lower()
                if kl in {"false", "0", "no"}:
                    false_v = float(v)
                elif kl in {"true", "1", "yes"}:
                    true_v = float(v)
            if true_v is not None:
                if false_v is None:
                    false_v = 1.0 - true_v
                arr = np.asarray([false_v, true_v], dtype=np.float64)
                if np.isfinite(arr).all() and arr.sum() > 0:
                    arr /= arr.sum()
                    return arr.tolist()
        if isinstance(soft, (int, float)):
            p1 = float(soft)
            if 0.0 <= p1 <= 1.0:
                return [1.0 - p1, p1]
        return None

    if isinstance(soft, dict):
        vals = []
        for key in source_keys:
            if key in soft:
                vals.append(float(soft[key]))
            elif str(key) in soft:
                vals.append(float(soft[str(key)]))
            else:
                vals.append(0.0)
        arr = np.asarray(vals, dtype=np.float64)
        if np.isfinite(arr).all() and arr.sum() > 0:
            arr /= arr.sum()
            return arr.tolist()
    return None


def process_jev_row(row, config_name: str, split_name: str, surface_seed: int):
    state = _parse_jsonish(row.get("state"))
    question = _parse_jsonish(row.get("question"))
    primitive = str(row.get("primitive") or (question.get("type") if isinstance(question, dict) else "") or "choice").lower()
    source_label = str(row.get("label"))
    row_id = str(row.get("id") or f"{config_name}:{split_name}:{_stable_int(row)}")
    instruction = _question_instruction(question)
    context = humanize_state(state)
    if not context:
        context = "No additional state was provided."

    metadata = {
        "source_hf_id": DATASET_NAME_DEFAULT,
        "source_config": config_name,
        "primitive": primitive,
        "domain": CONFIG_DOMAINS.get(config_name, "unknown"),
        "surface_seed": int(surface_seed),
    }

    if primitive == "choice":
        keys, descs = _source_options_from_question(question)
        if len(keys) < 2:
            raise ValueError(f"choice row has fewer than two criteria/options: id={row_id}")
        labels = _format_choice_labels(keys, descs)
        try:
            target_index = keys.index(source_label)
        except ValueError:
            # Rare datasets may stringify integral keys differently.
            normalized = [str(k).strip() for k in keys]
            target_index = normalized.index(str(source_label).strip())
        soft_probs = _soft_probs_from_source(row.get("soft_label"), primitive, keys)
        metadata.update({"label_surface": "key_description", "source_option_keys": keys})
        query = instruction

    elif primitive == "score":
        keys, descs = _source_options_from_question(question)
        if len(keys) < 2:
            raise ValueError(f"score row has fewer than two criteria/levels: id={row_id}")
        style_idx = _stable_int(config_name, split_name, row_id, surface_seed, "score-style", modulo=len(SCORE_SURFACE_STYLES))
        requested_style = SCORE_SURFACE_STYLES[style_idx]
        labels, actual_style = _format_score_labels(keys, descs, requested_style)
        try:
            target_index = keys.index(source_label)
        except ValueError:
            # README specifies score label is a level index. Fall back to integer index.
            idx = int(float(source_label))
            if not 0 <= idx < len(keys):
                raise
            target_index = idx
        soft_probs = _soft_probs_from_source(row.get("soft_label"), primitive, keys)
        metadata.update({
            "label_surface": actual_style,
            "source_option_keys": keys,
            "ordered": True,
            "score_values": list(range(len(keys))),
        })
        query = instruction

    elif primitive == "noul":
        gold_true = _normalize_noul_gold(source_label)
        pair_idx = _stable_int(config_name, split_name, row_id, surface_seed, "noul-style", modulo=len(NOUL_SURFACE_PAIRS))
        pos_surface, neg_surface = NOUL_SURFACE_PAIRS[pair_idx]

        # If the benchmark provides true/false criterion descriptions, retain them.
        true_desc = false_desc = ""
        if isinstance(question, dict):
            criteria = _parse_jsonish(question.get("criteria"))
            if isinstance(criteria, dict):
                for k, v in criteria.items():
                    kl = str(k).strip().lower()
                    if kl in {"true", "1", "yes"}:
                        true_desc = _criterion_description(v)
                    elif kl in {"false", "0", "no"}:
                        false_desc = _criterion_description(v)

        pos_label = f"{pos_surface} - {true_desc}" if true_desc else pos_surface
        neg_label = f"{neg_surface} - {false_desc}" if false_desc else neg_surface

        # Store either order with a deterministic 50/50 hash; the training collator also
        # independently randomizes order each epoch when enabled.
        reverse = bool(_stable_int(config_name, split_name, row_id, surface_seed, "noul-order", modulo=2))
        if reverse:
            labels = [neg_label, pos_label]
            target_index = 1 if gold_true else 0
            source_keys = ["false", "true"]
            soft_native = _soft_probs_from_source(row.get("soft_label"), primitive, ["false", "true"])
            soft_probs = soft_native if soft_native is None else [soft_native[0], soft_native[1]]
        else:
            labels = [pos_label, neg_label]
            target_index = 0 if gold_true else 1
            source_keys = ["true", "false"]
            soft_native = _soft_probs_from_source(row.get("soft_label"), primitive, ["false", "true"])
            soft_probs = soft_native if soft_native is None else [soft_native[1], soft_native[0]]

        metadata.update({
            "label_surface": f"{pos_surface}__{neg_surface}",
            "source_option_keys": source_keys,
            "gold_boolean": bool(gold_true),
        })
        query = _make_noul_query(instruction)

    else:
        raise ValueError(f"Unsupported primitive {primitive!r} for id={row_id}")

    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError(f"Invalid/duplicate processed labels for id={row_id}: {labels}")

    return {
        "context_id": row_id,
        "decision_id": row_id,
        "context": context,
        "query": query,
        "labels": labels,
        "target_index": int(target_index),
        "target_probs": None,
        "soft_target_probs": soft_probs,
        "source": config_name,
        "source_split": split_name,
        "source_config": config_name,
        "domain": CONFIG_DOMAINS.get(config_name, "unknown"),
        "task_family": config_name,
        "primitive": primitive,
        "label_style": metadata["label_surface"],
        "candidate_count": len(labels),
        "metadata": metadata,
    }
