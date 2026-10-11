#!/usr/bin/env python3
"""Run a trained decision model (hierarchical or BEV) from the checkpoint written by train.py.

Examples:
    python infer.py checkpoints_jev_hierarchical choice --context "Card was charged twice." \
        --query "Which team should handle this?" --choice billing --choice technical --choice sales
    python infer.py checkpoints_jev_hierarchical noul --context "Please refund me." \
        --query "Does this message explicitly request a refund?"
    python infer.py checkpoints_jev_bev score --context "Site is down for all users." \
        --query "How urgent is this?" --choice routine --choice elevated --choice critical
    python infer.py checkpoints_jev_hierarchical decide --request request.json   # object, or a list for a batch
    python infer.py checkpoints_jev_hierarchical batch --type noul --input items.jsonl --output out.jsonl

request.json (JEV-style):
    {"state": "Card was charged twice.",
     "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments or refunds", "technical": "Bugs or outages"}},
        "refund": {"type": "noul", "instructions": "Does this message explicitly request a refund?"}}}
"""

import argparse
import json
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="best_val directory, or the output dir containing it")
    p.add_argument("--device", default=None)
    p.add_argument("--model-dtype", choices=("auto", "float32", "bfloat16", "float16"), default=None)
    p.add_argument("--batch-size", type=int, default=32)
    sub = p.add_subparsers(dest="mode", required=True)

    for name in ("choice", "score", "noul"):
        s = sub.add_parser(name)
        s.add_argument("--context", default="")
        s.add_argument("--query", required=True)
        if name != "noul":
            s.add_argument("--choice", dest="choices", action="append", default=[],
                           help="Repeat for each option (for score: ordered low to high)")
    d = sub.add_parser("decide", help="JEV-style: state + questions JSON ('-' reads stdin)")
    d.add_argument("--request", required=True)
    b = sub.add_parser("batch", help="Many raw items from JSONL ('-' reads stdin); one JSON answer per line")
    b.add_argument("--type", dest="kind", choices=("choice", "score", "noul", "decide"), required=True)
    b.add_argument("--input", required=True)
    b.add_argument("--output", default=None, help="Write JSONL here instead of stdout")
    return p.parse_args(argv)


def _read(path):
    return sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()


def main():
    args = parse_args()
    from jevbench.inference import JevDecider

    decider = JevDecider.from_checkpoint(
        args.checkpoint, device=args.device, model_dtype=args.model_dtype, batch_size=args.batch_size
    )
    if args.mode == "choice":
        result = decider.choice(args.context, args.query, args.choices)
    elif args.mode == "score":
        result = decider.score(args.context, args.query, args.choices)
    elif args.mode == "noul":
        result = decider.noul(args.context, args.query)
    elif args.mode == "decide":
        request = json.loads(_read(args.request))
        if isinstance(request, list):
            result = decider.decide_batch(request)
        else:
            result = decider.decide(request.get("state"), request.get("questions"))
    else:
        items = [json.loads(line) for line in _read(args.input).splitlines() if line.strip()]
        run = {
            "choice": decider.choice_batch, "score": decider.score_batch,
            "noul": decider.noul_batch, "decide": decider.decide_batch,
        }[args.kind]
        lines = "\n".join(json.dumps(r) for r in run(items)) + "\n"
        if args.output:
            open(args.output, "w", encoding="utf-8").write(lines)
        else:
            sys.stdout.write(lines)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
