# JEV-Bench decision models

Train and run typed decision models (Choice, Score, Noul) on the JEV-Bench data pipeline.
Two architectures share the same data, training loop and inference API:

- `hierarchical`: shared-trunk branched Qwen + label/candidate Transformer
- `adaptive`: shared-trunk branched Qwen + decision-aware recurrent label reasoner
- `bev`: BEV packed masking/position trick + BEV choice head

Use the `py11` conda environment (`conda activate py11`).

## Train

```bash
python train.py hierarchical --full-run --epochs 1 --batch-size 16 --grad-accum-steps 8 --label-layers 2 --candidate-layers 0
python train.py adaptive --full-run --epochs 1 --batch-size 8 --grad-accum-steps 2 --label-layers 2 --max-loops 4
python train.py bev --full-run --epochs 3 --batch-size 1 --grad-accum-steps 16
python train.py bev --epochs 1 --batch-size 8 --grad-accum-steps 1 --no-wandb   # quick run
```

Adaptive decision feedback is enabled by default: each loop compares candidate states, revises cumulative
scores, and feeds its confidence/margin back into the next loop. Use `--no-decision-feedback` for the
branch-only loop baseline; `--context-cross-attn` additionally lets each loop read the question/context.

The best-validation checkpoint is written to `checkpoints_jev_<architecture>/best_val`
(`model.pt`, tokenizer files, `config.json`). Override the location with `--output-dir`.

## Infer

`infer.py` takes the checkpoint followed by a mode. The checkpoint is the `best_val` directory, the
output directory that contains it, or a Hugging Face repo id such as
`leobitz/decision-master-base-label-encode-0.1` (downloaded and cached on first use; the repo must
hold `model.pt`, `config.json` and the tokenizer files, as written by `train.py`). All modes accept `--device`, `--model-dtype` and `--batch-size` (default 32).

### Single questions

```bash
python infer.py checkpoints_jev_hierarchical choice \
    --context "Card was charged twice." --query "Which team should handle this?" \
    --choice billing --choice technical --choice sales

python infer.py checkpoints_jev_hierarchical noul \
    --context "Please refund me." --query "Does this message explicitly request a refund?"

# Score levels are ordered low to high
python infer.py checkpoints_jev_bev score \
    --context "Site is down for all users." --query "How urgent is this?" \
    --choice routine --choice elevated --choice critical
```

### JEV-style decisions (one shared state, many questions)

`request.json`:

```json
{
  "state": "Card was charged twice.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Payments or refunds", "technical": "Bugs or outages"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": {"routine": "Can wait", "elevated": "Same day", "critical": "Immediate"}
    },
    "refund": {"type": "noul", "instructions": "Does this message explicitly request a refund?"}
  }
}
```

```bash
python infer.py checkpoints_jev_hierarchical decide --request request.json
```

`state` may be text, a JSON object or an array. Output:

```json
{"answers": {
  "department": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.91, "technical": 0.09}, "confidence": 0.91},
  "urgency": {"type": "score", "score": 1.4, "legend": {"routine": "Can wait", "...": "..."}, "probabilities": {"...": 0.0}, "confidence": 0.55},
  "refund": {"type": "noul", "noul": 0.87}
}}
```

- `noul` is P(yes). It is averaged over every yes/no surface pair used in training
  (yes/no, true/false, correct/incorrect, affirmative/negative, supported/not supported, yes / true)
  in both orders, so the result does not depend on one wording.
- `score` is the probability-weighted level value (numeric criterion keys are used as values,
  otherwise 0-based level indices), averaged over the six training label styles.
- `choice` averages the original and reversed option order.
- `confidence` is the top probability.
- Invalid questions (unknown type, missing instructions, fewer than two options) raise an error.
  Treat that as an unavailable decision, not as a negative answer.

### Batches

Pass a JSON list to `decide`, or use `batch` with JSONL (one item per line, `-` reads stdin).
Output is one JSON answer per line, in input order. Every variant of every item is packed into
shared model batches, which is much faster than looping over single calls.

```bash
python infer.py checkpoints_jev_hierarchical batch --type noul --input items.jsonl --output answers.jsonl
```

Item shapes per `--type`:

| type | item |
| --- | --- |
| `choice` | `{"context": "...", "query": "...", "choices": ["a", "b"]}` (or `{key: description}`) |
| `score` | `{"context": "...", "query": "...", "levels": ["low", "mid", "high"]}` (or `{key: description}`) |
| `noul` | `{"context": "...", "query": "...", "criteria": {"true": "...", "false": "..."}}` (`criteria` optional) |
| `decide` | `{"state": ..., "questions": {...}}` |

## Python API

```python
from jevbench.inference import JevDecider

decider = JevDecider.from_checkpoint("checkpoints_jev_hierarchical", batch_size=32)

# Single
decider.choice(context, query, ["billing", "technical"])
decider.noul(context, query)
decider.score(context, query, ["routine", "elevated", "critical"])
decider.decide(state, questions)

# Batched: same answers as the single forms, returned as a list in input order
decider.choice_batch([{"context": c, "query": q, "choices": opts}, ...])
decider.noul_batch([{"context": c, "query": q}, ...])
decider.score_batch([{"context": c, "query": q, "levels": levels}, ...])
decider.decide_batch([{"state": s, "questions": questions}, ...])
```

Keep policy (authorization, thresholds, human review) in application code; use the returned
probabilities as signals and pick thresholds from a labeled evaluation set.
