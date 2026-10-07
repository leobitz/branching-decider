import unittest
import torch

from jevbench.data.bev_formatting import bev_generate_position_ids, bev_get_attention_mask


class BevFormattingTests(unittest.TestCase):
    def test_choice_positions_reset_and_choices_are_isolated(self):
        span = {"question_end_idx": 2, "choices_end_idx": [5, 9], "answer_end_idx": 11}
        pos = bev_generate_position_ids(span)
        self.assertEqual(pos[3], pos[6])  # both choices start at same logical position

        mask = bev_get_attention_mask(span)
        # Choice 1 [3:6] must not see choice 2 [6:10] and vice versa.
        self.assertEqual(int(mask[3:6, 6:10].sum().item()), 0)
        self.assertEqual(int(mask[6:10, 3:6].sum().item()), 0)
        # Answer slot begins at 10 and sees both choices.
        self.assertTrue(torch.all(mask[10:, 3:10] == 1))


if __name__ == "__main__":
    unittest.main()
