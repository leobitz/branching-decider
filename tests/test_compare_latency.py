import contextlib
import io
import json
import random
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from compare_latency import COMPARISON_KEYS, load_reports, print_comparison, summarize


ROOT = Path(__file__).resolve().parents[1]


def make_report(name="hierarchical_triton", timings=None):
    return {
        "schema_version": 1,
        "metadata": {key: "same" for key in COMPARISON_KEYS},
        "results": [{
            "name": name,
            "architecture": "bev" if name == "bev" else "hierarchical",
            "inference_backend": "standard_qwen_bev" if name == "bev" else "eager_fallback",
            "peak_cuda_allocated_mb": None,
            "accuracy": -1,
            "end_to_end_seconds": 0.1,
            "raw": {
                "batch_model_ms": [10.0, 30.0] if timings is None else timings,
                "batch_examples": [2, 1],
                "batch_correct": [1, 1],
                "batch_candidates": [4, 3],
            },
        }],
    }


class ComparisonTests(unittest.TestCase):
    def test_entry_points_select_only_their_architecture(self):
        shared = types.ModuleType("benchmark_test_latency")
        shared.main = Mock()
        with patch.dict(sys.modules, {"benchmark_test_latency": shared}):
            for architecture in ("hierarchical", "bev"):
                runpy.run_path(str(ROOT / f"benchmark_{architecture}.py"), run_name="__main__")
                shared.main.assert_called_once_with(architecture)
                shared.main.reset_mock()

    def test_summary_uses_raw_counts_and_timings(self):
        result = summarize(make_report()["results"][0])
        self.assertEqual(result["examples"], 3)
        self.assertEqual(result["batches"], 2)
        self.assertEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["model_ms_per_example"], 40 / 3)
        self.assertEqual(result["avg_candidates_per_example"], 7 / 3)
        self.assertEqual(result["batch_latency_ms_p50"], 20)
        self.assertEqual(result["batch_latency_ms_p95"], 29)
        self.assertAlmostEqual(result["batch_latency_ms_p99"], 29.8)
        self.assertAlmostEqual(result["end_to_end_ms_per_example"], 100 / 3)

    def test_single_batch_percentiles(self):
        result = make_report()["results"][0]
        result["raw"] = {key: values[:1] for key, values in result["raw"].items()}
        summary = summarize(result)
        for percentile in (50, 95, 99):
            self.assertEqual(summary[f"batch_latency_ms_p{percentile}"], 10)

    def test_invalid_raw_batches_are_rejected(self):
        for timings in ([], [10], [-1, 2], [float("nan"), 2], [float("inf"), 2]):
            with self.subTest(timings=timings), self.assertRaises(ValueError):
                summarize(make_report(timings=timings)["results"][0])

    def test_invalid_counts_are_rejected(self):
        for key, values in (("batch_examples", [0, 1]), ("batch_correct", [3, 1]),
                            ("batch_candidates", [1, 3]), ("batch_examples", [2.0, 1])):
            result = make_report()["results"][0]
            result["raw"][key] = values
            with self.subTest(key=key, values=values), self.assertRaises(ValueError):
                summarize(result)

    def test_speed_ratio_and_actual_backend_are_shown(self):
        output = io.StringIO()
        results = [summarize(make_report()["results"][0]),
                   summarize(make_report("bev", [20, 60])["results"][0])]
        with contextlib.redirect_stdout(output):
            print_comparison(results)
        self.assertIn("2.000x model-time", output.getvalue())
        self.assertIn("eager_fallback", output.getvalue())

    def write_reports(self, directory, second):
        paths = [Path(directory) / "hierarchical.json", Path(directory) / "bev.json"]
        for path, report in zip(paths, (make_report(), second)):
            path.write_text(json.dumps(report), encoding="utf-8")
        return paths

    def test_settings_mismatch_is_rejected_or_explicitly_allowed(self):
        second = make_report("bev")
        second["metadata"]["dataset_sha256"] = "different"
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_reports(directory, second)
            with self.assertRaisesRegex(ValueError, "dataset_sha256"):
                load_reports(paths)
            results, mismatches = load_reports(paths, allow_mismatch=True)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(mismatches), 1)

    def test_duplicate_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_reports(directory, make_report())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_reports(paths)

    def test_cli_saves_comparison_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_reports(directory, make_report("bev", [20, 60]))
            output = Path(directory) / "output" / "comparison.json"
            command = [sys.executable, "-S", str(ROOT / "compare_latency.py"),
                       *map(str, paths), "--output-json", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertIn("2.000x model-time", completed.stdout)
        self.assertEqual(len(saved["results"]), 2)
        self.assertEqual(saved["mismatches"], [])

    def test_cli_refuses_to_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_reports(directory, make_report("bev"))
            original = paths[0].read_text(encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-S", str(ROOT / "compare_latency.py"),
                 *map(str, paths), "--output-json", str(paths[0])],
                capture_output=True, text=True,
            )
            self.assertEqual(paths[0].read_text(encoding="utf-8"), original)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must not overwrite", completed.stderr)


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import benchmark_test_latency

        cls.benchmark = benchmark_test_latency

    def test_eager_has_a_separate_default_output(self):
        with patch.object(sys, "argv", ["benchmark_hierarchical.py", "--checkpoint", "checkpoint", "--eager"]):
            args = self.benchmark.parse_args("hierarchical")
        self.assertEqual(args.output_json, Path("hierarchical_eager_latency.json"))

    def test_parser_rejects_other_model_and_combined_runs(self):
        for architecture, extra in (("bev", "--hierarchical-checkpoint"),
                                    ("hierarchical", "--bev-checkpoint"),
                                    ("hierarchical", "--include-hierarchical-eager")):
            with self.subTest(architecture=architecture, extra=extra):
                with patch.object(sys, "argv", ["benchmark", "--checkpoint", "checkpoint", extra]), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.benchmark.parse_args(architecture)

    def test_dataset_order_does_not_depend_on_global_rng(self):
        cli = types.SimpleNamespace(seed=42, processed_data_dir=Path("cache"),
                                    max_test_examples=None, use_human_soft_labels=False)
        with patch.object(self.benchmark, "prepare_jev_bench_cache", return_value=Path("cache")), \
             patch.object(self.benchmark, "_read_jsonl", side_effect=lambda path: list(range(20))), \
             patch.object(self.benchmark, "JevBenchDecisionDataset", side_effect=lambda rows, *args, **kwargs: rows):
            _, first = self.benchmark.load_test_dataset({}, cli)
            with patch.object(random, "shuffle", side_effect=AssertionError("Global RNG used")):
                _, second = self.benchmark.load_test_dataset({}, cli)
        self.assertEqual(first, second)

    def test_main_runs_one_model_and_saves_metadata(self):
        for architecture in ("hierarchical", "bev"):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "report.json"
                name = "bev" if architecture == "bev" else "hierarchical_triton"
                dataset = MagicMock()
                dataset.rows = [{"candidate_count": 2}]
                dataset.__len__.return_value = 1
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(sys, "argv", ["benchmark", "--checkpoint", "checkpoint",
                                                                  "--output-json", str(output)]))
                    stack.enter_context(patch.object(self.benchmark, "seed_everything"))
                    stack.enter_context(patch.object(self.benchmark.torch.cuda, "is_available", return_value=False))
                    config_loader = stack.enter_context(patch.object(self.benchmark, "load_checkpoint_config", return_value={"architecture": architecture}))
                    stack.enter_context(patch.object(self.benchmark, "load_test_dataset", return_value=(Path(directory), dataset)))
                    runner = stack.enter_context(patch.object(self.benchmark, "benchmark_model", return_value=make_report(name)["results"][0]))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    self.benchmark.main(architecture)
                runner.assert_called_once()
                self.assertEqual(runner.call_args.args[1], architecture)
                config_loader.assert_called_once_with("checkpoint", architecture)
                saved = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(len(saved["results"]), 1)
                self.assertEqual(len(saved["metadata"]["dataset_sha256"]), 64)
                self.assertEqual(saved["metadata"]["seed"], 42)


if __name__ == "__main__":
    unittest.main()