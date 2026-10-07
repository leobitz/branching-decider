"""Architecture registry."""

from .hierarchical import DirectVariableChoiceQwenClassifier
from .bev import BEVQwenDecisionNetwork


def build_model(args, torch_dtype):
    if args.architecture == "hierarchical":
        return DirectVariableChoiceQwenClassifier(
            args.model_name,
            torch_dtype=torch_dtype,
            max_label_length=args.max_label_length,
            label_layers=args.label_layers,
            label_heads=args.label_heads,
            label_ff_mult=args.label_ff_mult,
            candidate_layers=args.candidate_layers,
            candidate_heads=args.candidate_heads,
            candidate_ff_mult=args.candidate_ff_mult,
            dropout=args.dropout,
        )
    if args.architecture == "bev":
        return BEVQwenDecisionNetwork(
            args.model_name,
            torch_dtype=torch_dtype,
            head_dim=args.head_dim,
            head_layers=args.head_layers,
        )
    raise ValueError(f"Unknown architecture: {args.architecture}")


def enable_gradient_checkpointing(model, architecture):
    if architecture == "hierarchical":
        model.qwen.gradient_checkpointing_enable()
    elif architecture == "bev":
        model.backbone.gradient_checkpointing_enable()
    else:
        raise ValueError(architecture)


def forward_batch(model, batch, architecture):
    if architecture == "hierarchical":
        return model(
            context_input_ids=batch["context_input_ids"],
            context_attention_mask=batch["context_attention_mask"],
            branch_input_ids=batch["branch_input_ids"],
            branch_attention_mask=batch["branch_attention_mask"],
            branch_to_example=batch["branch_to_example"],
            choice_index=batch["choice_index"],
            candidate_mask=batch["candidate_mask"],
        )
    if architecture == "bev":
        return model(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            attention_mask=batch["attention_mask"],
            choice_read_idx=batch["choice_read_idx"],
            choice_mask=batch["choice_mask"],
            answer_end_idx=batch["answer_end_idx"],
            task_type=batch["task_type"],
        )
    raise ValueError(architecture)
