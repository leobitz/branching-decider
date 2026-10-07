import json
import unittest

from jevbench.data.processing import process_jev_row


class ProcessingTests(unittest.TestCase):
    def test_choice_score_and_noul(self):
        base = {"state": json.dumps({"x": 1})}
        choice = process_jev_row({**base, "id":"c", "primitive":"choice", "question":json.dumps({"type":"choice","instructions":"Pick","criteria":{"a":"Alpha option","b":"Beta option"}}), "label":"b"}, "toy", "train", 42)
        self.assertEqual(choice["labels"][choice["target_index"]], "b - Beta option")

        score = process_jev_row({**base, "id":"s", "primitive":"score", "question":json.dumps({"type":"score","instructions":"Rate","criteria":["low","high"]}), "label":"1"}, "toy", "train", 42)
        self.assertEqual(score["target_index"], 1)

        noul = process_jev_row({**base, "id":"n", "primitive":"noul", "question":json.dumps({"type":"noul","instructions":"This is true."}), "label":"true"}, "toy", "train", 42)
        self.assertEqual(noul["candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
