import unittest

import torch
import torch.nn as nn

from jevbench.models.hierarchical_adaptive import (
    AdaptiveLoopQwenClassifier,
    adaptive_exit_loss,
    exit_distribution,
)


def make_model(max_loops=3, mode="joint", threshold=0.5, d=16, cross=False):
    """Heads-only model (no Qwen download) exercising the real forward paths."""
    model = AdaptiveLoopQwenClassifier.__new__(AdaptiveLoopQwenClassifier)
    nn.Module.__init__(model)
    model._init_heads(d, 5, 1, 4, 2, max_loops, 0.0, cross)
    model.exit_loss_mode = mode
    model.compute_weight, model.entropy_weight, model.aux_weight = 0.02, 0.01, 0.1
    model.gate_sharpness, model.exit_threshold = 10.0, threshold
    return model


def make_inputs(d=16):
    # 2 examples: 3 and 2 candidates -> 5 branches
    branch_to_example = torch.tensor([0, 0, 0, 1, 1])
    choice_index = torch.tensor([0, 1, 2, 0, 1])
    candidate_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    branches = torch.randn(5, 3, d)
    branch_mask = torch.ones(5, 3, dtype=torch.bool)
    branch_mask[4, 2] = False
    target = torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    return branches, branch_mask, branch_to_example, choice_index, candidate_mask, target


class ExitDistributionTests(unittest.TestCase):
    def test_sums_to_one_and_matches_example(self):
        q = torch.tensor([[0.20, 0.50, 0.25]])
        p = exit_distribution(q)
        torch.testing.assert_close(p, torch.tensor([[0.20, 0.40, 0.10, 0.30]]))
        torch.testing.assert_close(p.sum(-1), torch.ones(1))

    def test_joint_loss_matches_manual_value(self):
        losses = torch.tensor([[1.2, 0.7, 0.45, 0.4]])
        exit_logits = torch.logit(torch.tensor([[0.20, 0.50, 0.25]]))
        loss, terms, p = adaptive_exit_loss(losses, exit_logits, "joint", 0.0, 0.0, 0.0)
        self.assertAlmostEqual(loss.item(), 0.685, places=4)
        self.assertAlmostEqual(terms["expected_loops"].item(), 0.2 + 0.8 + 0.3 + 1.2, places=4)

    def test_gate_loss_only_trains_exit_logits(self):
        losses = torch.tensor([[1.0, 0.5, 0.49]], requires_grad=True)
        exit_logits = torch.zeros(1, 2, requires_grad=True)
        loss, _, _ = adaptive_exit_loss(losses, exit_logits, "gate", compute_weight=0.05)
        loss.backward()
        self.assertIsNone(losses.grad)
        self.assertIsNotNone(exit_logits.grad)
        # Big improvement at loop 1 -> keep going (exit prob pushed down); none at loop 2 -> exit.
        self.assertGreater(exit_logits.grad[0, 0].item(), 0)
        self.assertLess(exit_logits.grad[0, 1].item(), 0)


class AdaptiveModelTests(unittest.TestCase):
    def test_loop_prefix_consistency(self):
        torch.manual_seed(0)
        branches, bm, *_ = make_inputs()
        reasoner = make_model().label_reasoner.eval()
        full = reasoner(branches, bm)
        self.assertEqual(full.shape, (5, 3, 16))
        torch.testing.assert_close(reasoner(branches, bm, num_loops=2), full[:, :2])

    def test_training_loss_backward_reaches_blocks_and_exit_head(self):
        torch.manual_seed(1)
        model = make_model().train()
        branches, bm, b2e, ci, cmask, target = make_inputs()
        out = model._forward_fixed(branches, bm, b2e, ci, cmask, target, None)
        torch.testing.assert_close(out["exit_probabilities"].sum(-1), torch.ones(2))
        torch.testing.assert_close(out["probabilities"].sum(-1), torch.ones(2))
        out["loss"].backward()
        self.assertGreater(model.exit_head[1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.label_reasoner.blocks[0].ff[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.label_reasoner.loop_embedding.grad.abs().sum().item(), 0)

    def test_gate_mode_only_updates_exit_head(self):
        torch.manual_seed(2)
        model = make_model(mode="gate").train()
        branches, bm, b2e, ci, cmask, target = make_inputs()
        out = model._forward_fixed(branches, bm, b2e, ci, cmask, target, None)
        out["loss"].backward()
        self.assertGreater(model.exit_head[1].weight.grad.abs().sum().item(), 0)
        for name, p in model.named_parameters():
            if not name.startswith(("exit_head", "exit_decision_projection")):
                self.assertIsNone(p.grad, name)

    def test_decisions_are_revised_and_fed_back(self):
        torch.manual_seed(9)
        branches, bm, b2e, ci, cmask, target = make_inputs()
        model = make_model().train()
        with torch.no_grad():
            model.decision_feedback[-1].weight.normal_(0, 0.1)
            model.candidate_gate.fill_(0.5)
        out = model._forward_fixed(branches, bm, b2e, ci, cmask, target, None)
        self.assertEqual(out["step_logits"].shape, (2, 3, 3))
        out["loss"].backward()
        self.assertGreater(model.decision_feedback[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.candidate_gate.grad.abs().sum().item(), 0)

    def test_decision_feedback_can_be_disabled(self):
        torch.manual_seed(10)
        branches, bm, b2e, ci, cmask, target = make_inputs()
        model = make_model().eval()
        with torch.no_grad():
            model.decision_feedback[-1].weight.normal_(0, 0.1)
        with torch.no_grad():
            enabled = model._forward_fixed(branches, bm, b2e, ci, cmask, None, None)["logits"]
            model.decision_feedback_enabled = False
            disabled = model._forward_fixed(branches, bm, b2e, ci, cmask, None, None)["logits"]
        self.assertFalse(torch.allclose(enabled, disabled, atol=1e-5))

    def test_early_exit_matches_fixed_depth_at_chosen_depth(self):
        torch.manual_seed(3)
        branches, bm, b2e, ci, cmask, target = make_inputs()
        for threshold, expected_depth in ((0.0, 1), (1.1, 3)):
            model = make_model(threshold=threshold).eval()
            out = model._forward_early_exit(branches, bm, b2e, ci, cmask)
            self.assertTrue((out["exit_depth"] == expected_depth).all())
            ref = model._forward_fixed(branches, bm, b2e, ci, cmask, None, expected_depth)
            torch.testing.assert_close(out["probabilities"], ref["probabilities"], atol=1e-5, rtol=1e-5)

    def test_mixed_exit_depths_match_fixed_depth_per_example(self):
        torch.manual_seed(4)
        branches, bm, b2e, ci, cmask, target = make_inputs()
        model = make_model(threshold=0.5).eval()
        with torch.no_grad():
            model.exit_head[1].weight.normal_(0, 3.0)
            out = model._forward_early_exit(branches, bm, b2e, ci, cmask)
            for ex in range(2):
                depth = int(out["exit_depth"][ex])
                ref = model._forward_fixed(branches, bm, b2e, ci, cmask, None, depth)
                torch.testing.assert_close(
                    out["probabilities"][ex], ref["probabilities"][ex], atol=1e-5, rtol=1e-5
                )


class ContextCrossAttentionTests(unittest.TestCase):
    def make_context(self, d=16):
        ctx = torch.randn(2, 4, d)
        ctx_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        return ctx, ctx_mask

    def test_zero_gate_matches_plain_reasoner(self):
        torch.manual_seed(5)
        branches, bm, b2e, *_ = make_inputs()
        plain = make_model().label_reasoner.eval()
        cross = make_model(cross=True).label_reasoner.eval()
        missing = cross.load_state_dict(plain.state_dict(), strict=False)
        self.assertTrue(all("cross_attn" in k for k in missing.missing_keys))
        ctx, ctx_mask = self.make_context()
        torch.testing.assert_close(
            cross(branches, bm, context_states=ctx, context_attention_mask=ctx_mask, branch_to_example=b2e),
            plain(branches, bm),
        )

    def test_context_changes_output_and_ignores_padding(self):
        torch.manual_seed(6)
        branches, bm, b2e, *_ = make_inputs()
        reasoner = make_model(cross=True).label_reasoner.eval()
        with torch.no_grad():
            for block in reasoner.blocks:
                block.cross_attn.gate.fill_(1.0)
        ctx, ctx_mask = self.make_context()
        run = lambda c: reasoner(
            branches, bm, context_states=c, context_attention_mask=ctx_mask, branch_to_example=b2e
        )
        base = run(ctx)
        self.assertEqual(base.shape, (5, 3, 16))
        self.assertFalse(torch.allclose(base, run(torch.randn_like(ctx)), atol=1e-4))
        padded = ctx.clone()
        padded[1, 2:] = 100.0
        torch.testing.assert_close(run(padded)[3:], base[3:], atol=1e-5, rtol=1e-5)

    def test_early_exit_with_context_matches_fixed_depth(self):
        torch.manual_seed(7)
        branches, bm, b2e, ci, cmask, target = make_inputs()
        ctx = self.make_context()
        model = make_model(threshold=0.5, cross=True).eval()
        with torch.no_grad():
            model.exit_head[1].weight.normal_(0, 3.0)
            for block in model.label_reasoner.blocks:
                block.cross_attn.gate.fill_(1.0)
            out = model._forward_early_exit(branches, bm, b2e, ci, cmask, ctx)
            for ex in range(2):
                depth = int(out["exit_depth"][ex])
                ref = model._forward_fixed(branches, bm, b2e, ci, cmask, None, depth, ctx)
                torch.testing.assert_close(
                    out["probabilities"][ex], ref["probabilities"][ex], atol=1e-5, rtol=1e-5
                )

    def test_training_backward_reaches_cross_attention(self):
        torch.manual_seed(8)
        branches, bm, b2e, ci, cmask, target = make_inputs()
        ctx, ctx_mask = self.make_context()
        ctx.requires_grad_(True)
        model = make_model(cross=True).train()
        out = model._forward_fixed(branches, bm, b2e, ci, cmask, target, None, (ctx, ctx_mask))
        out["loss"].backward()
        cross = model.label_reasoner.blocks[0].cross_attn
        self.assertGreater(cross.gate.grad.abs().sum().item(), 0)
        self.assertIsNotNone(ctx.grad)


if __name__ == "__main__":
    unittest.main()
