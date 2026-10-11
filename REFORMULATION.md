# JEV-Bench: parallel question reformulation

This is a **data-generation-only addition**. The ordinary `train.py` data loader,
training loop, models, CLI and baseline dataset download behavior are unchanged.
The new script and `jevbench/reformulation/` package are opt-in.

## Main command

From the repository root, install the existing dependencies (`pip install -r requirements.txt`) and run:

```bash
python reformulate_dataset.py --num-generated 50000 --workers 8 \
  --output-dir jev_bench_processed_reformulated
```

By default the script downloads the repository's existing upstream dataset,
`Praveenrajus/jev-bench`, and discovers **all configurations**. It retains all
successfully processed original rows, generates 50,000 **additional** rows in
parallel, and saves the combined dataset. It does not replace or modify
upstream labels.

Other useful generation budgets:

```bash
# New examples = valid originals, independently for every source/split.
python reformulate_dataset.py --ratio 1.0 --workers 8

# Exactly 2000 generated examples per source/config, proportionally across splits.
python reformulate_dataset.py --per-source 2000 --workers 8

# Exactly 2000 generated examples per source AND split.
python reformulate_dataset.py --per-source-split 2000 --workers 8 

# 2500 additional training examples across just MNLI and MMLU.
python reformulate_dataset.py --configs mnli,mmlu --splits train \
  --num-generated 2500 --workers 4

# Also save the intermediate raw JEV rows, including both original and synthetic.
python reformulate_dataset.py --configs mmlu --num-generated 3000 --save-raw
```

If no generation amount is specified, it defaults to `--ratio 1.0`, meaning
one generated row per valid original row in each config/split (approximately
doubling the original data).

The global `--num-generated` budget is allocated proportionally to the number
of usable upstream rows in the selected configs/splits, using largest-remainder
rounding, so that the **sum of the allocated quotas equals the exact request**.
The other exact count options allocate per source/config or per source+split.
Requests that cannot be satisfied with distinct valid outputs fail visibly by
default (override with `--allow-shortfall` or `--allow-duplicates`).

## Output format

The script writes **the identical processed cache layout and ordinary training
fields** that the repository already uses:

```text
jev_bench_processed_reformulated/
  train.jsonl
  validation.jsonl
  test.jsonl
  manifest.json
  generation_errors.jsonl       # only if errors were recorded
  raw/                          # only with --save-raw
    mmlu/
      train.jsonl
      validation.jsonl
      test.jsonl
    mnli/
      ...
```

`train.jsonl`, etc. contain the same `context`, `query`, `labels`,
`target_index`, `soft_target_probs`, `source`, `primitive`, and other fields from
`jevbench/data/processing.py`. Both new columns are **top-level JSON fields**:

- Every original row: `"original": true`, `"generation": null`.
- Every generated row: `"original": false`, `"generation": { ... }`.

Example (unrelated fields omitted for brevity):

```json
{
  "context": "...",
  "query": "Do the two NLI pairs have the same relation?",
  "labels": ["same", "different"],
  "target_index": 0,
  "original": false,
  "generation": {
    "strategy": "mnli_same_relation_pair",
    "description": "Present two premise-hypothesis pairs and ask whether the two pairs have the same NLI relation.",
    "target_rule": "Positive iff gold_relation_1 == gold_relation_2.",
    "row_usage": "2",
    "source": "mnli",
    "split": "train",
    "parent_ids": ["...", "..."],
    "parent_labels": ["entailment", "entailment"],
    "canonical_target": "TRUE",
    "answerability": true,
    "gold_offered": true,
    "seed": 42,
    "sample_slot": 23,
    "raw_id": "aug:mnli:train:...",
    "details": {}
  }
}
```

A generated row may change its **primitive** (for example `noul` or `score`
becomes a two-option or multichoice `choice`), but it is always passed through
the project's existing `process_jev_row()` function. The `generation` object
is a genuine JSON dictionary, not an escaped string.

The manifest has `original_split_counts`, `generated_split_counts`,
`split_counts`, `primitive_counts`, `strategy_counts`, and a per-config/split
request-versus-achieved summary. It includes all the original manifest keys,
so the original loader can consume the reformulated cache without changes.

## Use in normal training without changing a single model file

To train with the generated dataset, point the existing CLI at the saved cache:

```bash
python train.py hierarchical --processed-data-dir jev_bench_processed_reformulated
# or
python train.py bev --processed-data-dir jev_bench_processed_reformulated
```

The existing loader will reuse the saved cache provided the HF config list,
revision, seed, and processing version match. Avoid `--force-reprocess` during
training or the existing loader will intentionally rebuild its old-style cache.
If you selected fewer configs, pass exactly those `--configs` to training.

**Evaluation caveat:** Augmenting validation/test makes a different benchmark;
for a strict untouched official evaluation set, use `--splits train` so
validation and test remain originals only.

## Strategy coverage

`jevbench/reformulation/sources/` has **22 readable Python modules** with
**295 individually named strategy functions**. Every function includes a
before/after example, target rule, and row-selection requirements in its
docstring. The accompanying `strategy_plan.json` provides a machine-readable
summary. Examples of implemented strategy families:

- candidate correctness as true/false, yes/no, correct/incorrect, etc.
- candidate subset membership, partitions and opaque temporary label mappings
- `OTHER` when the original semantic answer was removed from offered labels
- `UNKNOWN` when the required input/evidence is unavailable
- joint `OTHER` + `UNKNOWN` + ordinary-answer discrimination
- two-pair MNLI same-relation, first/second/both/neither and logical combinations
- same-class pairs, reference matching, odd-one-out, and pairwise comparisons
- score thresholds, intervals, coarsened bands, ordinal comparisons, and differences
- human vote share thresholds/comparisons and exact soft-target mappings when valid
- evidence sufficiency and irrelevant-context distractors

Surface pools include 23 paired binary variants, 32 `OTHER` variants,
37 `UNKNOWN` variants, and alternative `BOTH/NEITHER` phrasings. Paired labels
are kept semantically compatible (for example, `same/different` only for
relation matching, not arbitrary question-answer truth values).

When gold information is available but omitted from the label set, the target
is `OTHER`. When decisive **supplied evidence** is omitted, the target is
`UNKNOWN`. An item is *not* labelled unknown simply because humans disagree.
Partial ablations (where some facts are removed but the rest may still answer
the question) are **disabled by default** until a trusted semantic validator is
provided; the deterministic whole-input-removal strategies work automatically.

For partial-ablation strategies, a validator may be supplied as an importable
`module:function` argument to the CLI. The validator receives the proposed
transformed raw row followed by the original raw row, and returns a boolean.
Never return `True` without actually checking answer insufficiency.

## How multiprocessing works

The driver first loads one source/split at a time and stages only valid original
rows, then assigns generation quotas. It creates a separate process pool for
each source/split and initializes workers once with that split's read-only
indexed pool. Each worker chooses a source-specific strategy, one or more
same-split rows, output-label surfaces, and deterministic target labels. Results
are processed through the existing row converter and merged. It bounds
in-flight tasks in batches rather than submitting millions of jobs at once.

This keeps `--workers 1` useful for debugging and scales to more processes
without crossing the train/validation/test boundary. Parallel generation is
deterministic for a fixed seed/worker-independent sample index.

## Testing

```bash
PYTHONPATH=.:tests pytest -q tests/test_reformulation.py
```

These offline tests exercise all 295 generators, validate the special labels,
check quota math, and run a two-process merged-data build using a fake HF
loader. They do not require model weights or a network connection.

`jevbench/reformulation/examples.jsonl` contains **290 fully generated before/after
examples** produced by the current implementation. The remaining five
validator-gated strategies are documented in their function docstrings but do
not have automatically approved synthetic targets without an external
insufficiency validator.
