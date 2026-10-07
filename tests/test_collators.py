import unittest
from types import SimpleNamespace

from jevbench.data.collators import (
    BEVJevCollator,
    BranchingVariableChoiceCollator,
    build_collators,
)


class CollatorConstructionTests(unittest.TestCase):
    def test_build_collators_initializes_both_architectures(self):
        tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0, padding_side="left")
        options = dict(
            seed=42,
            randomize_candidate_order=True,
            max_length=128,
            max_label_length=16,
            max_state_tokens=128,
            max_choice_tokens=16,
            max_packed_tokens=512,
        )
        for architecture, collator_type in (
            ("hierarchical", BranchingVariableChoiceCollator),
            ("bev", BEVJevCollator),
        ):
            with self.subTest(architecture=architecture):
                train_collator, eval_collator = build_collators(
                    SimpleNamespace(architecture=architecture, **options), tokenizer
                )
                self.assertIsInstance(train_collator, collator_type)
                self.assertIsInstance(eval_collator, collator_type)
                self.assertTrue(train_collator.training)
                self.assertFalse(eval_collator.training)
                self.assertEqual(train_collator.seed, 43)
                self.assertEqual(eval_collator.seed, 44)
                self.assertEqual(tokenizer.padding_side, "right")


if __name__ == "__main__":
    unittest.main()