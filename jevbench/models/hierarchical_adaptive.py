"""Hierarchical Qwen decision model with an adaptive-depth (looped) label reasoner.

Qwen3 runs exactly once (``encoded``).  The compute that is made adaptive is the
:class:`~jevbench.models.hierarchical.LabelTokenReasoner` level: one weight-tied stack of
Transformer blocks is applied repeatedly to ``[CLS | label tokens]`` and the
model learns *how many loops* to spend per example.

After every loop ``t`` (``t = 1..T``):

    y_t = scorer(CLS_t)                      candidate logits
    q_t = sigmoid(exit_head(CLS_t))          P(exit at t | reached t), pooled over the example's candidates
    p_t = q_t * prod_{j<t} (1 - q_j)         actual exit distribution, with q_T forced to 1

The exit decision is made per *example* (the mean of its candidates' exit logits) so every
candidate of an example is scored at the same depth, in training and at inference.

Loss modes (``exit_loss_mode``):

* ``joint``: ``sum_t p_t L_t + lambda*E[T] - beta*H(p) + eta*mean_t L_t``
* ``aux``:   ``mean_t L_t`` only (stage 1 of the two-stage recipe; exit head untrained)
* ``gate``:  stage 2.  BCE of the exit head against ``sigmoid(k*(L_t - L_{t+1} - lambda))`` computed
  from detached losses; the exit head input is detached so only the exit head receives gradients.

Inference (``eval``, no early-exit override) stops each example at the first loop whose exit
probability reaches ``exit_threshold`` and drops finished examples from later loops.

Optional context cross-attention (``context_cross_attn``): each loop's blocks also cross-attend to the
question/context states from Qwen (keys/values only), so the output stays ``[CLS | label tokens]``-shaped.
It is off by default (plain thinking over the branch tokens only).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchical import InferenceOptimizedQwen3BranchedModel, TransformerBlock

EXIT_LOSS_MODES = ("joint", "aux", "gate")
_NEG = -1e9


def exit_distribution(exit_probs: torch.Tensor) -> torch.Tensor:
    """Conditional exit probs ``[B, T-1]`` -> actual exit distribution ``[B, T]`` (sums to 1)."""
    remaining = torch.ones_like(exit_probs[:, 0]) if exit_probs.shape[1] else exit_probs.new_ones(exit_probs.shape[0])
    probs = []
    for t in range(exit_probs.shape[1]):
        probs.append(remaining * exit_probs[:, t])
        remaining = remaining * (1.0 - exit_probs[:, t])
    probs.append(remaining)
    return torch.stack(probs, dim=1)


def adaptive_exit_loss(
    step_losses: torch.Tensor,
    exit_logits: torch.Tensor,
    mode: str = "joint",
    compute_weight: float = 0.02,
    entropy_weight: float = 0.01,
    aux_weight: float = 0.1,
    gate_sharpness: float = 10.0,
):
    """step_losses ``[B, T]`` per-loop CE, exit_logits ``[B, T-1]``. Returns ``(loss, terms, p)``."""
    if mode not in EXIT_LOSS_MODES:
        raise ValueError(f"exit_loss_mode must be one of {EXIT_LOSS_MODES}, got {mode!r}")
    B, T = step_losses.shape
    q = torch.sigmoid(exit_logits.float())
    p = exit_distribution(q)
    depths = torch.arange(1, T + 1, device=p.device, dtype=p.dtype)
    expected_loops = (p * depths).sum(-1).mean()
    entropy = -(p * p.clamp_min(1e-8).log()).sum(-1).mean()
    aux = step_losses.mean()
    prediction = (p * step_losses).sum(-1).mean()

    if mode == "joint":
        loss = prediction + compute_weight * expected_loops - entropy_weight * entropy + aux_weight * aux
    elif mode == "aux":
        loss = aux
    else:
        with torch.no_grad():
            improvement = step_losses[:, :-1] - step_losses[:, 1:]
            keep_going = torch.sigmoid(gate_sharpness * (improvement - compute_weight))
        # P(continue) = 1 - q_t = sigmoid(-exit_logit)
        loss = F.binary_cross_entropy_with_logits(-exit_logits.float(), keep_going)

    terms = {
        "prediction": prediction.detach(),
        "aux": aux.detach(),
        "expected_loops": expected_loops.detach(),
        "entropy": entropy.detach(),
    }
    return loss, terms, p


class ContextCrossAttention(nn.Module):
    """Branch tokens attend to their example's question/context states (keys/values only).

    The output shape equals the query shape, so the branch layout is unchanged.  The tanh gate starts
    at zero, so enabling the module initially leaves the plain looped reasoner's function intact.
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Parameter(torch.zeros(1))

    def _split(self, t):
        B, S, D = t.shape
        return t.view(B, S, self.num_heads, D // self.num_heads).transpose(1, 2)

    def project_context(self, context_states):
        """``[B, P, D]`` -> per-example keys/values ``[B, H, P, Dh]``; reusable across loops."""
        c = self.norm_ctx(context_states)
        return self._split(self.k_proj(c)), self._split(self.v_proj(c))

    def forward(self, x, context_kv, context_mask, example_index):
        """x ``[M, S, D]``; ``example_index [M]`` selects each branch's example in the context tensors."""
        k, v = context_kv
        q = self._split(self.q_proj(self.norm_q(x)))
        attn_mask = context_mask[example_index][:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k[example_index], v[example_index], attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(x.shape)
        return torch.tanh(self.gate).to(out.dtype) * self.o_proj(out)


class ContextTransformerBlock(TransformerBlock):
    """Pre-norm block: self-attention, then cross-attention to the context, then feed-forward."""

    def __init__(self, d_model, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__(d_model, num_heads, ff_mult, dropout)
        self.cross_attn = ContextCrossAttention(d_model, num_heads, dropout)

    def forward(self, x, valid_mask, context_kv, context_mask, example_index):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=~valid_mask, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.cross_attn(x, context_kv, context_mask, example_index)
        x = x + self.ff(self.norm2(x))
        return x * valid_mask.unsqueeze(-1).to(x.dtype)


class AdaptiveLabelTokenReasoner(nn.Module):
    """Weight-tied looped Transformer over ``[CLS | label tokens]`` with per-loop CLS readouts.

    With ``context_cross_attn=True`` every loop's blocks can also read the question/context states.
    """

    def __init__(
        self, d_model, max_label_length, max_loops=4, num_layers=2, num_heads=8, ff_mult=4, dropout=0.1,
        context_cross_attn=False,
    ):
        super().__init__()
        if max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if num_layers < 1:
            raise ValueError("AdaptiveLabelTokenReasoner needs at least one layer")
        self.max_loops = int(max_loops)
        self.max_label_length = int(max_label_length)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.empty(1, self.max_label_length + 1, d_model))
        self.loop_embedding = nn.Parameter(torch.empty(self.max_loops, d_model))
        self.context_cross_attn = bool(context_cross_attn)
        block_cls = ContextTransformerBlock if self.context_cross_attn else TransformerBlock
        self.blocks = nn.ModuleList([block_cls(d_model, num_heads, ff_mult, dropout) for _ in range(num_layers)])
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.loop_embedding, mean=0.0, std=0.02)

    def prepare(self, branch_states, branch_attention_mask):
        """Build the initial loop state ``x [M,1+L,D]`` and its validity mask ``[M,1+L]``."""
        if branch_states.ndim != 3:
            raise ValueError(f"branch_states must be [M,L,D], got {tuple(branch_states.shape)}")
        if branch_states.shape[:2] != branch_attention_mask.shape:
            raise ValueError("branch state/mask shapes do not agree")
        M, L, D = branch_states.shape
        if L > self.max_label_length:
            raise ValueError(
                f"Observed label length {L} exceeds configured max_label_length={self.max_label_length}"
            )
        cls = self.cls_token.to(dtype=branch_states.dtype).expand(M, 1, D)
        x = torch.cat([cls, branch_states], dim=1)
        cls_mask = torch.ones(M, 1, dtype=torch.bool, device=branch_attention_mask.device)
        valid = torch.cat([cls_mask, branch_attention_mask.to(torch.bool)], dim=1)
        x = x + self.position_embedding[:, : L + 1].to(dtype=x.dtype, device=x.device)
        return x * valid.unsqueeze(-1).to(x.dtype), valid

    def prepare_context(self, context_states, context_attention_mask):
        """Project the context once per example (shared by all loops). ``None`` when cross-attention is off."""
        if not self.context_cross_attn:
            return None
        if context_states is None or context_attention_mask is None:
            raise ValueError("context_cross_attn=True requires context states and mask")
        mask = context_attention_mask.to(torch.bool).clone()
        mask[:, 0] |= ~mask.any(-1)  # an empty context must not produce an all-masked softmax
        kvs = [block.cross_attn.project_context(context_states) for block in self.blocks]
        return kvs, mask

    def step(self, x, valid, loop_index, context=None, example_index=None):
        """One application of the shared block stack.

        ``context`` comes from :meth:`prepare_context`; ``example_index [M]`` maps each row of ``x`` to
        its example in the context tensors.
        """
        x = x + self.loop_embedding[loop_index].to(dtype=x.dtype)
        x = x * valid.unsqueeze(-1).to(x.dtype)
        if self.context_cross_attn:
            kvs, ctx_mask = context
            for block, kv in zip(self.blocks, kvs):
                x = block(x, valid, kv, ctx_mask, example_index)
        else:
            for block in self.blocks:
                x, _ = block(x, valid)
        return x

    def forward(
        self, branch_states, branch_attention_mask, num_loops=None,
        context_states=None, context_attention_mask=None, branch_to_example=None,
    ):
        """Run ``num_loops`` (default ``max_loops``) loops; returns CLS readouts ``[M, T, D]``."""
        T = self.max_loops if num_loops is None else int(num_loops)
        if not 1 <= T <= self.max_loops:
            raise ValueError(f"num_loops must be in [1, {self.max_loops}], got {num_loops}")
        x, valid = self.prepare(branch_states, branch_attention_mask)
        context = self.prepare_context(context_states, context_attention_mask)
        if self.context_cross_attn and branch_to_example is None:
            raise ValueError("context_cross_attn=True requires branch_to_example")
        readouts = []
        for t in range(T):
            x = self.step(x, valid, t, context, branch_to_example)
            readouts.append(x[:, 0])
        return torch.stack(readouts, dim=1)


class AdaptiveLoopQwenClassifier(nn.Module):
    """Branched Qwen3 -> decision-aware looped reasoner -> scorer, with a learned exit head."""

    def __init__(
        self,
        model_name="Qwen/Qwen3-0.6B-Base",
        torch_dtype=None,
        max_label_length=96,
        label_layers=2,
        label_heads=8,
        label_ff_mult=4,
        max_loops=4,
        dropout=0.1,
        use_inference_kernel: bool = True,
        exit_loss_mode="joint",
        compute_weight=0.002,
        entropy_weight=0.05,
        aux_weight=0.5,
        gate_sharpness=10.0,
        exit_threshold=0.7,
        context_cross_attn=False,
        decision_feedback=True,
    ):
        super().__init__()
        if exit_loss_mode not in EXIT_LOSS_MODES:
            raise ValueError(f"exit_loss_mode must be one of {EXIT_LOSS_MODES}")
        self.qwen = InferenceOptimizedQwen3BranchedModel.from_pretrained(model_name, dtype=torch_dtype)
        self.qwen.set_inference_kernel(use_inference_kernel)
        d = self.qwen.config.hidden_size
        self._init_heads(
            d, max_label_length, label_layers, label_heads, label_ff_mult, max_loops, dropout,
            context_cross_attn, decision_feedback,
        )
        self.exit_loss_mode = exit_loss_mode
        self.compute_weight = compute_weight
        self.entropy_weight = entropy_weight
        self.aux_weight = aux_weight
        self.gate_sharpness = gate_sharpness
        self.exit_threshold = exit_threshold

    def _init_heads(
        self, d, max_label_length, layers, heads, ff_mult, max_loops, dropout,
        context_cross_attn=False, decision_feedback=True,
    ):
        self.label_reasoner = AdaptiveLabelTokenReasoner(
            d_model=d,
            max_label_length=max_label_length,
            max_loops=max_loops,
            num_layers=layers,
            num_heads=heads,
            ff_mult=ff_mult,
            dropout=dropout,
            context_cross_attn=context_cross_attn,
        )
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.exit_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.decision_feedback_enabled = bool(decision_feedback)
        self.decision_feedback = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.candidate_reasoner = TransformerBlock(d, heads, ff_mult, dropout)
        self.candidate_gate = nn.Parameter(torch.zeros(1))
        self.exit_decision_projection = nn.Linear(6, d, bias=False)
        nn.init.zeros_(self.decision_feedback[-1].weight)
        nn.init.zeros_(self.decision_feedback[-1].bias)
        nn.init.zeros_(self.exit_decision_projection.weight)

    @property
    def max_loops(self) -> int:
        return self.label_reasoner.max_loops

    @property
    def inference_backend(self) -> str:
        return self.qwen.inference_backend

    def set_inference_kernel(self, enabled: bool = True) -> None:
        self.qwen.set_inference_kernel(enabled)

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _example_mean(values, branch_to_example, num_examples):
        """Mean of per-branch ``values [M, ...]`` over each example's branches -> ``[B, ...]``."""
        sums = values.new_zeros((num_examples,) + values.shape[1:]).index_add_(0, branch_to_example, values)
        counts = torch.bincount(branch_to_example, minlength=num_examples).clamp_min(1)
        return sums / counts.view(-1, *([1] * (values.ndim - 1))).to(values.dtype)

    def _exit_logits(self, readouts, decision_features, detach=False):
        """Return exit logits ``[M, T-1]``; the final loop is forced to exit."""
        states = readouts[:, :-1]
        features = decision_features[:, :-1]
        if detach:
            states, features = states.detach(), features.detach()
        if self.decision_feedback_enabled:
            states = states + self.exit_decision_projection(features.to(states.dtype))
        return self.exit_head(states).squeeze(-1)

    def _candidate_exchange(self, cls, branch_to_example, choice_index, candidate_mask):
        """Let candidates compare CLS states while keeping one CLS state per branch."""
        examples = torch.unique(branch_to_example, sorted=True)
        local_example = torch.searchsorted(examples, branch_to_example)
        mask = candidate_mask[examples].to(torch.bool)
        if not mask.any(-1).all().item():
            raise ValueError("Every active example must have at least one candidate")
        grid = cls.new_zeros((len(examples), candidate_mask.shape[1], cls.shape[-1]))
        grid[local_example, choice_index] = cls
        updated, _ = self.candidate_reasoner(grid, mask)
        gate = torch.tanh(self.candidate_gate).to(cls.dtype)
        return (grid + gate * (updated - grid))[local_example, choice_index]

    @staticmethod
    def _decision_features(
        probabilities, candidate_mask, branch_to_example, choice_index,
        previous_probabilities=None, previous_top=None,
    ):
        """Build branch-aligned confidence, rival-margin, change, and entropy features."""
        probabilities = probabilities.float()
        own = probabilities[branch_to_example, choice_index]
        row_probs = probabilities[branch_to_example]
        other_mask = candidate_mask[branch_to_example].to(torch.bool).clone()
        other_mask.scatter_(1, choice_index[:, None], False)
        rival = row_probs.masked_fill(~other_mask, -1.0).max(-1).values.clamp_min(0.0)
        margin = own - rival

        top = probabilities.argmax(-1)
        is_top = (top[branch_to_example] == choice_index).to(probabilities.dtype)
        if previous_probabilities is None:
            delta = torch.zeros_like(own)
            changed = torch.zeros_like(own)
        else:
            delta = own - previous_probabilities[branch_to_example, choice_index]
            changed = (top[branch_to_example] != previous_top[branch_to_example]).to(probabilities.dtype)

        entropy = -(probabilities * probabilities.clamp_min(1e-12).log() * candidate_mask).sum(-1)
        counts = candidate_mask.sum(-1).to(probabilities.dtype)
        entropy = torch.where(
            counts > 1, entropy / counts.log().clamp_min(1e-12), torch.zeros_like(entropy)
        )
        return torch.stack(
            (own, margin, is_top, delta, entropy[branch_to_example], changed), dim=-1
        ), top

    @staticmethod
    def _inject_feedback(x, feedback):
        return torch.cat((x[:, :1] + feedback[:, None, :], x[:, 1:]), dim=1)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        context_input_ids,
        context_attention_mask,
        branch_input_ids,
        branch_attention_mask,
        branch_to_example,
        choice_index,
        candidate_mask,
        target_probs=None,
        num_loops: Optional[int] = None,
        adaptive_inference: bool = True,
        return_attention=False,
    ):
        """Training (or ``num_loops``/``adaptive_inference=False``) runs a fixed number of loops.

        Eval without those overrides stops each example at its learned exit depth.  Pass
        ``target_probs`` while training to get ``out["loss"]``.
        """
        if return_attention:
            raise NotImplementedError("attention maps are not supported by the adaptive reasoner")

        encoded = self.qwen.forward_branches(
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            branch_input_ids=branch_input_ids,
            branch_attention_mask=branch_attention_mask,
            branch_to_example=branch_to_example,
        )
        branch_states = encoded.branch_last_hidden_state
        context = (encoded.context_last_hidden_state, context_attention_mask)

        early_exit = (not self.training) and adaptive_inference and num_loops is None
        if early_exit:
            out = self._forward_early_exit(
                branch_states, branch_attention_mask,
                branch_to_example, choice_index, candidate_mask, context,
            )
        else:
            out = self._forward_fixed(
                branch_states, branch_attention_mask,
                branch_to_example, choice_index, candidate_mask, target_probs, num_loops, context,
            )
        out["context_states"] = encoded.context_last_hidden_state
        out["inference_backend"] = self.inference_backend
        return out

    def _scatter_candidates(self, flat, branch_to_example, choice_index, candidate_mask):
        """flat ``[M, ...]`` -> ``[B, N, ...]`` filled with a large negative number at padded slots."""
        B, N = candidate_mask.shape
        out = flat.new_full((B, N) + flat.shape[1:], _NEG)
        out[branch_to_example, choice_index] = flat
        return out

    def _forward_fixed(
        self, branch_states, branch_mask,
        branch_to_example, choice_index, candidate_mask, target_probs, num_loops, context=None,
    ):
        B = candidate_mask.shape[0]
        reasoner = self.label_reasoner
        T = reasoner.max_loops if num_loops is None else int(num_loops)
        if not 1 <= T <= reasoner.max_loops:
            raise ValueError(f"num_loops must be in [1, {reasoner.max_loops}], got {num_loops}")
        ctx_states, ctx_mask = context if context is not None else (None, None)
        context_kv = reasoner.prepare_context(ctx_states, ctx_mask)
        x, valid = reasoner.prepare(branch_states, branch_mask)

        readouts, logits_by_loop, features_by_loop = [], [], []
        previous_scores = None
        previous_probabilities = previous_top = None
        for t in range(T):
            x = reasoner.step(x, valid, t, context_kv, branch_to_example)
            cls = x[:, 0]
            if self.decision_feedback_enabled:
                cls = self._candidate_exchange(cls, branch_to_example, choice_index, candidate_mask)
                x = torch.cat((cls[:, None], x[:, 1:]), dim=1)

            delta_scores = self.scorer(cls).squeeze(-1).float()
            scores = delta_scores if previous_scores is None or not self.decision_feedback_enabled \
                else previous_scores + delta_scores
            logits_b = self._scatter_candidates(scores, branch_to_example, choice_index, candidate_mask)
            logits_b = logits_b.masked_fill(~candidate_mask, _NEG)
            probabilities = F.softmax(logits_b, dim=-1)
            features, top = self._decision_features(
                probabilities, candidate_mask, branch_to_example, choice_index,
                previous_probabilities, previous_top,
            )
            readouts.append(cls)
            logits_by_loop.append(logits_b)
            features_by_loop.append(features)

            if self.decision_feedback_enabled and t + 1 < T:
                x = self._inject_feedback(x, self.decision_feedback(features.to(cls.dtype)))
                previous_scores = scores
                previous_probabilities, previous_top = probabilities, top

        readouts = torch.stack(readouts, dim=1)
        decision_features = torch.stack(features_by_loop, dim=1)
        logits_bnt = torch.stack(logits_by_loop, dim=1)
        step_log_probs = F.log_softmax(logits_bnt.float(), dim=-1)  # [B, T, N]
        step_probs = step_log_probs.exp()

        out = {"step_logits": logits_bnt.permute(0, 2, 1)}
        use_exit = T > 1
        if use_exit:
            detach = self.exit_loss_mode == "gate" and self.training
            exit_logits = self._example_mean(
                self._exit_logits(readouts, decision_features, detach=detach).float(), branch_to_example, B
            )
            p = exit_distribution(torch.sigmoid(exit_logits))
        else:
            exit_logits = None
            p = step_probs.new_ones(B, 1)

        # Expected-exit mixture while training; the deepest executed loop for fixed-depth evaluation.
        if self.training and use_exit:
            probs = (p.unsqueeze(-1) * step_probs).sum(1)
        else:
            probs = step_probs[:, -1]
        probs = probs.masked_fill(~candidate_mask, 0.0)
        out["probabilities"] = probs
        out["logits"] = probs.clamp_min(1e-12).log().masked_fill(~candidate_mask, _NEG)
        out["exit_probabilities"] = p.detach()
        depths = torch.arange(1, T + 1, device=p.device, dtype=p.dtype)
        out["expected_loops"] = (p.detach() * depths).sum(-1).mean()

        if target_probs is not None and self.training:
            target = target_probs.to(step_log_probs.dtype)
            step_losses = -(target.unsqueeze(1) * step_log_probs).sum(-1)  # [B, T]
            if use_exit:
                loss, terms, _ = adaptive_exit_loss(
                    step_losses, exit_logits, self.exit_loss_mode,
                    self.compute_weight, self.entropy_weight, self.aux_weight, self.gate_sharpness,
                )
            else:
                loss, terms = step_losses.mean(), {"aux": step_losses.mean().detach()}
            out["loss"] = loss
            out["loss_terms"] = terms
            out["step_losses"] = step_losses.detach()
        return out

    @torch.no_grad()
    def _forward_early_exit(
        self, branch_states, branch_mask,
        branch_to_example, choice_index, candidate_mask, context=None,
    ):
        B, N = candidate_mask.shape
        reasoner = self.label_reasoner
        T = reasoner.max_loops
        x, valid = reasoner.prepare(branch_states, branch_mask)
        ctx_states, ctx_mask = context if context is not None else (None, None)
        ctx = reasoner.prepare_context(ctx_states, ctx_mask)

        logits = x.new_full((B, N), _NEG, dtype=torch.float32)
        exit_depth = torch.zeros(B, dtype=torch.long, device=x.device)
        active = torch.arange(x.shape[0], device=x.device)  # branch ids still looping
        previous_scores = previous_probabilities = previous_top = None

        for t in range(T):
            ex = branch_to_example[active]
            examples = torch.unique(ex, sorted=True)
            local_ex = torch.searchsorted(examples, ex)
            local_mask = candidate_mask[examples]
            x = reasoner.step(x, valid, t, ctx, ex)
            cls = x[:, 0]
            if self.decision_feedback_enabled:
                cls = self._candidate_exchange(cls, ex, choice_index[active], candidate_mask)
                x = torch.cat((cls[:, None], x[:, 1:]), dim=1)

            delta_scores = self.scorer(cls).squeeze(-1).float()
            branch_logits = delta_scores if previous_scores is None or not self.decision_feedback_enabled \
                else previous_scores + delta_scores
            logits_active = branch_logits.new_full((len(examples), N), _NEG)
            logits_active[local_ex, choice_index[active]] = branch_logits
            logits_active = logits_active.masked_fill(~local_mask, _NEG)
            probabilities = F.softmax(logits_active, dim=-1)
            features, top = self._decision_features(
                probabilities, local_mask, local_ex, choice_index[active],
                previous_probabilities, previous_top,
            )
            if self.decision_feedback_enabled and t + 1 < T:
                x = self._inject_feedback(x, self.decision_feedback(features.to(cls.dtype)))

            if t == T - 1:
                stop_branch = torch.ones_like(active, dtype=torch.bool)
            else:
                exit_state = cls
                if self.decision_feedback_enabled:
                    exit_state = exit_state + self.exit_decision_projection(features.to(exit_state.dtype))
                exit_logits = self.exit_head(exit_state).squeeze(-1).float()
                q = torch.sigmoid(self._example_mean(exit_logits, local_ex, len(examples)))
                stop_branch = (q >= self.exit_threshold)[local_ex]
            done = active[stop_branch]
            logits[branch_to_example[done], choice_index[done]] = branch_logits[stop_branch]
            exit_depth[ex[stop_branch]] = t + 1
            if bool(stop_branch.all()):
                break
            keep = ~stop_branch
            next_active = active[keep]
            next_examples = torch.unique(branch_to_example[next_active], sorted=True)
            next_local = torch.searchsorted(examples, next_examples)
            if self.decision_feedback_enabled:
                previous_scores = branch_logits[keep]
                previous_probabilities, previous_top = probabilities[next_local], top[next_local]
            active, x, valid = next_active, x[keep], valid[keep]

        logits = logits.masked_fill(~candidate_mask, _NEG)
        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
            "exit_depth": exit_depth,
            "expected_loops": exit_depth.float().mean(),
        }


__all__ = [
    "AdaptiveLabelTokenReasoner",
    "AdaptiveLoopQwenClassifier",
    "EXIT_LOSS_MODES",
    "adaptive_exit_loss",
    "exit_distribution",
]
