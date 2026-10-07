"""User's hierarchical shared-trunk branched-Qwen decision model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen_branching import Qwen3BranchedModel, gather_last_branch_state

class TransformerBlock(nn.Module):
    """Standard bidirectional Transformer block with an explicit padding mask."""

    def __init__(self, d_model, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, valid_mask, return_attention=False):
        # valid_mask: [batch_like, sequence_length], True for real tokens/items.
        key_padding_mask = ~valid_mask
        attn_out, attn_weights = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.drop1(attn_out))
        x = self.norm2(x + self.ff(x))
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        return x, attn_weights


class LabelTokenReasoner(nn.Module):
    """K-layer reasoner over [shared learned CLS] + candidate LABEL tokens only.

    The candidate token states supplied by Qwen are already conditioned on the shared
    context/query.  Therefore these top layers never repeat the long input sequence.

    Input:
        branch_states: [sum(M_b), L_label, D]
        branch_attention_mask: [sum(M_b), L_label]
    Output:
        one CLS vector per real candidate: [sum(M_b), D]
    """

    def __init__(self, d_model, max_label_length, num_layers=2, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.num_layers = int(num_layers)
        self.max_label_length = int(max_label_length)

        # Exactly one CLS parameter is shared by every label sequence and every example.
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        # The Qwen branch states are already position-sensitive, but a small top-level
        # learned position embedding gives the newly inserted CLS token an explicit position
        # and makes token order directly available to the bidirectional top Transformer.
        self.position_embedding = nn.Parameter(torch.empty(1, self.max_label_length + 1, d_model))
        self.input_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_mult, dropout)
            for _ in range(self.num_layers)
        ])
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, branch_states, branch_attention_mask, return_attention=False):
        if branch_states.ndim != 3:
            raise ValueError(f"branch_states must be [M,L,D], got {tuple(branch_states.shape)}")
        if branch_attention_mask.ndim != 2:
            raise ValueError("branch_attention_mask must be [M,L]")
        if branch_states.shape[:2] != branch_attention_mask.shape:
            raise ValueError("branch state/mask shapes do not agree")

        # K=0 is a useful ablation that exactly recovers the previous fixed-width
        # candidate representation: the last valid Qwen branch state (normally EOS).
        if self.num_layers == 0:
            return gather_last_branch_state(branch_states, branch_attention_mask), []

        M, L, D = branch_states.shape
        if L > self.max_label_length:
            raise ValueError(
                f"Observed label length {L} exceeds configured max_label_length={self.max_label_length}"
            )

        cls = self.cls_token.to(dtype=branch_states.dtype).expand(M, 1, D)
        x = torch.cat([cls, branch_states], dim=1)
        cls_mask = torch.ones(M, 1, dtype=torch.bool, device=branch_attention_mask.device)
        valid_mask = torch.cat([cls_mask, branch_attention_mask.to(torch.bool)], dim=1)

        pos = self.position_embedding[:, : L + 1].to(dtype=x.dtype, device=x.device)
        x = self.input_norm(x + pos)
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        attention_maps = []
        for block in self.blocks:
            x, weights = block(x, valid_mask, return_attention=return_attention)
            if return_attention:
                attention_maps.append(weights)
        return x[:, 0], attention_maps


class DirectVariableChoiceQwenClassifier(nn.Module):
    """Branched Qwen3 -> K label-token layers -> L candidate-CLS layers -> scorer.

    Stage 1 is cheap with long inputs because it processes ONLY label-branch tokens.
    Stage 2 sees one vector per candidate and is permutation equivariant because no
    candidate-index positional embeddings are added.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen3-0.6B-Base",
        torch_dtype=None,
        max_label_length=96,
        label_layers=2,
        label_heads=8,
        label_ff_mult=4,
        candidate_layers=2,
        candidate_heads=8,
        candidate_ff_mult=4,
        dropout=0.1,
    ):
        super().__init__()
        self.qwen = Qwen3BranchedModel.from_pretrained(model_name, dtype=torch_dtype)
        d = self.qwen.config.hidden_size

        self.label_reasoner = LabelTokenReasoner(
            d_model=d,
            max_label_length=max_label_length,
            num_layers=label_layers,
            num_heads=label_heads,
            ff_mult=label_ff_mult,
            dropout=dropout,
        )

        self.pre_candidate_norm = nn.LayerNorm(d)
        self.candidate_blocks = nn.ModuleList([
            TransformerBlock(d, candidate_heads, candidate_ff_mult, dropout)
            for _ in range(candidate_layers)
        ])
        self.scorer = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
        )

    def forward(
        self,
        context_input_ids,
        context_attention_mask,
        branch_input_ids,
        branch_attention_mask,
        branch_to_example,
        choice_index,
        candidate_mask,
        return_attention=False,
        disable_candidate_transformer=False,
    ):
        encoded = self.qwen.forward_branches(
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            branch_input_ids=branch_input_ids,
            branch_attention_mask=branch_attention_mask,
            branch_to_example=branch_to_example,
        )

        # Qwen returns only the branch/token states here.  Each label token has already
        # causally attended to the shared input/query in every Qwen layer.  The K new
        # Transformer layers therefore operate on [CLS] + LABEL TOKENS ONLY.
        flat_candidate_states, label_attention_maps = self.label_reasoner(
            encoded.branch_last_hidden_state,
            branch_attention_mask,
            return_attention=return_attention,
        )

        # Regroup M candidate CLS states back to [B, N_max, D].
        B, N = candidate_mask.shape
        D = flat_candidate_states.shape[-1]
        candidate_states = flat_candidate_states.new_zeros(B, N, D)
        candidate_states[branch_to_example, choice_index] = flat_candidate_states
        candidate_states = self.pre_candidate_norm(candidate_states)
        candidate_states = candidate_states * candidate_mask.unsqueeze(-1).to(candidate_states.dtype)

        # L Transformer layers across candidate CLS vectors.  No candidate positional
        # embeddings are added, so permuting labels permutes outputs in the same way.
        candidate_attention_maps = []
        x = candidate_states
        if not disable_candidate_transformer:
            for block in self.candidate_blocks:
                x, weights = block(x, candidate_mask, return_attention=return_attention)
                if return_attention:
                    candidate_attention_maps.append(weights)

        logits = self.scorer(x).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, -1e9)
        probabilities = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "flat_candidate_states": flat_candidate_states,
            "candidate_states_before_set": candidate_states,  # compatibility name
            "candidate_states_after_set": x,                  # compatibility name
            "context_states": encoded.context_last_hidden_state,
            "label_attention_maps": label_attention_maps,
            "candidate_attention_maps": candidate_attention_maps,
            "attention_maps": candidate_attention_maps,       # compatibility name
        }
