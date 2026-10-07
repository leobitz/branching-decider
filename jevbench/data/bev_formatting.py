"""BEV input formatting, position IDs, and block-causal attention mask.

Adapted from https://github.com/avbiswas/bev-train/tokenization.py.
"""

from __future__ import annotations

import torch

from .constants import TASK_PROMPTS

def bev_truncate(tokenizer, text, max_tokens):
    tokens = tokenizer.encode(str(text))
    if len(tokens) <= max_tokens:
        return str(text)
    return tokenizer.decode(tokens[:max_tokens])


def bev_build_question(tokenizer, state, instructions, task_type, max_state_tokens):
    # Exact ordering from bev-train/tokenization.py.
    return (
        f"{TASK_PROMPTS[task_type]}\n"
        f"Question: {instructions}\n\n"
        f"{bev_truncate(tokenizer, state, max_state_tokens)}\n"
    )


def bev_tokenize_prompt(tokenizer, question, choices):
    choice_begin_token = tokenizer.encode("<option>")
    choice_end_token = tokenizer.encode("</option>")
    answer_prompt = "The answer is:"

    question_tokens = tokenizer.encode(question)
    answer_tokens = tokenizer.encode(answer_prompt)
    choice_tokens = [tokenizer.encode(choice) for choice in choices]

    final_tokens = []
    final_tokens.extend(question_tokens)
    span = {
        "question_end_idx": len(question_tokens) - 1,
        "choices_end_idx": [],
        "answer_end_idx": 0,
    }

    for choice in choice_tokens:
        final_tokens.extend(choice_begin_token)
        final_tokens.extend(choice)
        final_tokens.extend(choice_end_token)
        span["choices_end_idx"].append(len(final_tokens) - 1)

    final_tokens.extend(answer_tokens)
    span["answer_end_idx"] = len(final_tokens) - 1
    return final_tokens, span


def bev_generate_position_ids(span):
    position_ids = []

    start_idx = 0
    for _ in range(span["question_end_idx"] + 1):
        position_ids.append(start_idx)
        start_idx += 1

    next_choice_begin_idx = start_idx
    for choice_end_idx in span["choices_end_idx"]:
        # BEV's key trick: start_idx does NOT advance between choices.
        for idx in range(choice_end_idx - next_choice_begin_idx + 1):
            position_ids.append(start_idx + idx)
        next_choice_begin_idx = choice_end_idx + 1

    # Answer begins after the longest logical choice position.
    answer_start_idx = max(position_ids) + 1
    for idx in range(span["answer_end_idx"] + 1 - len(position_ids)):
        position_ids.append(answer_start_idx + idx)
    return position_ids


def bev_get_attention_mask(span):
    L = span["answer_end_idx"] + 1
    attention_mask = torch.zeros((L, L))

    question_len = span["question_end_idx"] + 1
    attention_mask[:question_len, :question_len] = torch.tril(torch.ones(question_len, question_len))

    choice_spans = []
    choice_start_idx = question_len
    for choice_end_idx in span["choices_end_idx"]:
        choice_spans.append((choice_start_idx, choice_end_idx))
        choice_start_idx = choice_end_idx + 1

    # Each choice is causal only inside itself.
    for choice_start_idx, choice_end_idx in choice_spans:
        n = choice_end_idx - choice_start_idx + 1
        attention_mask[
            choice_start_idx:choice_end_idx + 1,
            choice_start_idx:choice_end_idx + 1,
        ] = torch.tril(torch.ones(n, n))

    # Every choice can see the full question/state prefix.
    attention_mask[question_len:, :question_len] = 1

    # Answer slot sees everything physically before it, including all choices.
    answer_tokens_begin = choice_spans[-1][1] + 1
    attention_mask[answer_tokens_begin:, :answer_tokens_begin] = 1
    answer_n = L - answer_tokens_begin
    attention_mask[answer_tokens_begin:, answer_tokens_begin:] = torch.tril(torch.ones(answer_n, answer_n))
    return attention_mask


def bev_encode_example(tokenizer, state, instructions, choices, task_type, max_state_tokens, max_choice_tokens):
    question = bev_build_question(tokenizer, state, instructions, task_type, max_state_tokens)
    choices = [bev_truncate(tokenizer, choice, max_choice_tokens) for choice in choices]

    final_tokens, span = bev_tokenize_prompt(tokenizer, question, choices)
    position_ids = bev_generate_position_ids(span)
    attention_mask = bev_get_attention_mask(span)

    # Exact BEV readout: final content token, immediately before </option>.
    choice_end_token_len = len(tokenizer.encode("</option>"))
    choice_read_idx = [end_idx - choice_end_token_len for end_idx in span["choices_end_idx"]]

    return {
        "input_ids": final_tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "choice_read_idx": choice_read_idx,
        "answer_end_idx": span["answer_end_idx"],
    }
