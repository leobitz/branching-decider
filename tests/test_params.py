import unittest
import torch.nn as nn

from jevbench.utils.params import count_parameters


class ParamTests(unittest.TestCase):
    def test_counts(self):
        m = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 1))
        total, trainable = count_parameters(m)
        self.assertEqual(total, trainable)
        self.assertEqual(total, 21)


if __name__ == "__main__":
    unittest.main()
