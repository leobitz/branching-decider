"""Shared full-finetuning engine for both architectures."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from jevbench.data.collators import build_collators
from jevbench.data.dataset import load_jev_bench_splits
from jevbench.models import build_model, enable_gradient_checkpointing, forward_batch
from jevbench.training.checkpoint import BestCheckpointManager
from jevbench.training.metrics import (
    argmax_accuracy, distribution_cross_entropy, evaluate, move_tensor_batch, raise_nonfinite_forward,
)
from jevbench.training.sampler import BalancedGroupSampler
from jevbench.utils.params import print_parameter_summary
from jevbench.utils.runtime import resolve_model_dtype, seed_everything


def _model_config(args, processed_dir, parameter_summary):
    skip = {"hf_cache_dir"}
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in skip}
    config["processed_data_dir"] = str(processed_dir)
    config["parameter_summary"] = parameter_summary
    return config


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
                config={**vars(args), "device": device.type},
            )
        except Exception as exc:
            print(f"W&B disabled ({exc}). Continuing without logging.")

    def log(payload, step):
        if wandb_run is not None:
            wandb_run.log(payload, step=step)

    processed_dir, datasets = load_jev_bench_splits(args)
    train_data, val_data, test_data = datasets["train"], datasets["validation"], datasets["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_collator, eval_collator = build_collators(args, tokenizer)

    if args.balanced_train_sampling:
        draws = args.train_samples_per_epoch or len(train_data)
        train_sampler = BalancedGroupSampler(
            train_data.rows, group_field=args.balance_field, num_samples=draws, seed=args.seed + 100
        )
        print(
            f"Balanced training by {args.balance_field!r}: "
            f"{len(train_sampler.group_names)} groups, {len(train_sampler):,} draws/epoch"
        )
    else:
        train_sampler = None
        print(f"Balanced sampling disabled: {len(train_data):,} examples/epoch")

    print(f"DataLoader workers: {args.num_workers}")
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler,
        collate_fn=train_collator, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_data, batch_size=max(1, args.batch_size), shuffle=False, collate_fn=eval_collator,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_data, batch_size=max(1, args.batch_size), shuffle=False, collate_fn=eval_collator,
        num_workers=args.num_workers,
    )

    model_dtype = resolve_model_dtype(device, args.model_dtype)
    print("model dtype:", model_dtype)
    model = build_model(args, model_dtype).to(device)

    # Both experiments are full-finetuning experiments; BEV intentionally does not use LoRA/freeze.
    model.requires_grad_(True)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model, args.architecture)
        print("gradient checkpointing: enabled")

    parameter_summary = print_parameter_summary(model, args.architecture)
    log({f"params/{k}": v for k, v in parameter_summary.items()}, step=0)

    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_update_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_update_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    checkpoint = BestCheckpointManager(args.output_dir, "best_val", "max")
    model_config = _model_config(args, processed_dir, parameter_summary)
    last_eval_step = None

    def run_validation(step):
        nonlocal last_eval_step
        if last_eval_step == step:
            return None
        metrics = evaluate(
            model, val_loader, device, use_amp, amp_dtype, args.architecture,
            max_batches=args.eval_max_batches,
        )
        log({f"val/{k}": v for k, v in metrics.items()}, step)
        saved = checkpoint.maybe_save(metrics["acc"], metrics["loss"], step, model, tokenizer, model_config)
        tag = " -> saved best_val" if saved is not None else ""
        print(
            f"[step {step}] val loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
            f"nll={metrics['nll']:.4f} brier={metrics['brier']:.4f}{tag}"
        )
        last_eval_step = step
        model.train()
        return metrics

    n_original = sum(1 for r in train_data.rows if r["original"])
    print(
        f"Training data: original={n_original:,} reformulated={len(train_data) - n_original:,} "
        f"total={len(train_data):,}"
    )

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    running_loss, running_acc = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"{args.architecture} epoch {epoch}/{args.epochs}")
        for micro_step, batch in enumerate(pbar, 1):
            b = move_tensor_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = forward_batch(model, b, args.architecture)
                loss = distribution_cross_entropy(out["logits"], b["target_probs"], b["candidate_mask"])
                scaled_loss = loss / args.grad_accum_steps

            raise_nonfinite_forward(out, loss, micro_step)
            scaler.scale(scaled_loss).backward()
            running_loss.append(float(loss.item()))
            running_acc.append(argmax_accuracy(out["logits"].detach(), b["target_probs"]))

            if micro_step % args.grad_accum_steps == 0 or micro_step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.log_every == 0:
                    tr_loss = float(np.mean(running_loss[-args.grad_accum_steps:]))
                    tr_acc = float(np.mean(running_acc[-args.grad_accum_steps:]))
                    lr = float(scheduler.get_last_lr()[0])
                    log({"train/loss": tr_loss, "train/acc": tr_acc, "train/lr": lr, "epoch": epoch}, global_step)
                    pbar.set_postfix(loss=f"{tr_loss:.3f}", acc=f"{tr_acc:.3f}")

                if args.eval_every and global_step % args.eval_every == 0:
                    run_validation(global_step)

        run_validation(global_step)

    print("\nBest validation checkpoint:")
    if checkpoint.best_score is not None:
        print(
            f"  acc={checkpoint.best_score:.4f} loss={checkpoint.best_loss:.4f} "
            f"step={checkpoint.best_step}  {checkpoint.path}"
        )
        state = torch.load(checkpoint.path / "model.pt", map_location=device)
        model.load_state_dict(state)
    else:
        print("  no validation checkpoint was saved; evaluating current model")

    test_metrics = evaluate(model, test_loader, device, use_amp, amp_dtype, args.architecture, max_batches=None)
    print(
        f"\nTEST (best validation model): loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['acc']:.4f} nll={test_metrics['nll']:.4f} "
        f"brier={test_metrics['brier']:.4f}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    log({f"test/{k}": v for k, v in test_metrics.items()}, global_step)

    if wandb_run is not None:
        wandb_run.finish()
    return test_metrics
