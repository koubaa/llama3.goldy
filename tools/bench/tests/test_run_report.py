from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

TOOLS = pathlib.Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench import common, report, schema  # noqa: E402
from bench.adapters.goldy.run import bench_cmd  # noqa: E402
from bench.run import MODELS, matrix  # noqa: E402


def _row(**overrides):
    obj = schema.base_result(
        engine="goldy",
        execution="native",
        checkpoint={
            "path": "models/stories15M.bin",
            "sha256": common.CHECKPOINT_SHA256["stories15M.bin"],
            "config": {
                "dim": 288,
                "hidden_dim": 768,
                "n_layers": 6,
                "n_heads": 6,
                "n_kv_heads": 6,
                "vocab_size": 32000,
                "max_seq_len": 256,
            },
        },
        workload={
            "tier": "compatibility",
            "prompt": common.DREAM_PROMPT,
            "batch": 1,
            "context_len": 50,
            "total_positions": 50,
            "decode_steps": 44,
            "sampling": "greedy",
        },
        tokens={
            "prompt": [1, 76],
            "generated": [13],
            "text": common.DREAM_STORY,
            "match_expected": True,
        },
        phases={
            "load_s": 0.2,
            "warmup_s": 0.1,
            "prompt_s": 0.01,
            "ttft_s": 0.012,
            "decode_step_s": [0.001, 0.002],
            "compat_elapsed_s": 0.02,
        },
        metrics={
            "prompt_tok_s": 400.0,
            "decode_tok_s": 800.0,
            "legacy_compat_tok_s": 2000.0,
        },
        replay_stats={"records": 2, "topology_records": 0, "clean_submits": 40, "resubmit_hits": 40},
    )
    obj.update(overrides)
    return obj


class MatrixTests(unittest.TestCase):
    def test_smoke_is_compat_plus_one_scale(self):
        jobs = matrix(engines=["goldy", "pytorch-eager"], models=["15m"], contexts=[8, 32], smoke=True)
        modes = {(j["engine"], j["mode"], j["context"]) for j in jobs}
        self.assertIn(("goldy", "compatibility", None), modes)
        self.assertIn(("goldy", "scaling", 8), modes)
        self.assertNotIn(("goldy", "scaling", 32), modes)

    def test_full_covers_models_and_contexts(self):
        jobs = matrix(
            engines=["goldy"],
            models=list(MODELS),
            contexts=list(common.SCALING_CONTEXT_LENGTHS),
            smoke=False,
        )
        scaling = [j for j in jobs if j["mode"] == "scaling"]
        self.assertEqual(len(scaling), 3 * len(common.SCALING_CONTEXT_LENGTHS))
        self.assertTrue(any(j["mode"] == "compatibility" for j in jobs))

    def test_llama_cpp_bench_uses_bench_mode(self):
        jobs = matrix(engines=["llama.cpp-bench"], models=["15m"], contexts=[8], smoke=True)
        self.assertTrue(all(j["mode"] == "bench" for j in jobs))


class GoldyAdapterTests(unittest.TestCase):
    def test_bench_cmd_flags(self):
        cmd = " ".join(
            bench_cmd(
                pathlib.Path("llama3-goldy-bench"),
                checkpoint=pathlib.Path("models/stories15M.bin"),
                tokenizer=pathlib.Path("models/tokenizer.bin"),
                mode="scaling",
                prompt=common.DREAM_PROMPT,
                context=32,
                decode_steps=16,
                total_positions=50,
                warmups=1,
                reps=3,
            )
        )
        self.assertIn("--mode scaling", cmd)
        self.assertIn("--context 32", cmd)
        self.assertIn("--reps 3", cmd)


class ReportTests(unittest.TestCase):
    def test_writes_markdown_and_json(self):
        rows = [
            _row(),
            _row(
                metrics={
                    "prompt_tok_s": 420.0,
                    "decode_tok_s": 820.0,
                    "legacy_compat_tok_s": 2100.0,
                }
            ),
            _row(
                engine="pytorch-eager",
                replay_stats=None,
                metrics={
                    "prompt_tok_s": 100.0,
                    "decode_tok_s": 200.0,
                    "legacy_compat_tok_s": 300.0,
                },
            ),
        ]
        scale = _row()
        scale["workload"] = {
            "tier": "scaling",
            "prompt": common.DREAM_PROMPT,
            "batch": 1,
            "context_len": 8,
            "total_positions": 24,
            "decode_steps": 16,
            "sampling": "greedy",
        }
        scale["tokens"]["match_expected"] = True
        rows.append(scale)
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            md, summary = report.write_report(
                rows,
                metadata={"hostname": "test", "gpu": "fake", "captured_at": "now"},
                skipped=["llama3.cuda: WSL not on PATH"],
                out_md=out / "baseline.md",
                out_json=out / "baseline.json",
            )
            text = md.read_text(encoding="utf-8")
            self.assertIn("Compatibility", text)
            self.assertIn("goldy", text)
            self.assertIn("pytorch-eager", text)
            self.assertIn("Scaling decode tok/s", text)
            self.assertIn("llama3.cuda: WSL not on PATH", text)
            self.assertIn("Execution: goldy=native", text)
            data = json.loads((out / "baseline.json").read_text(encoding="utf-8"))
            self.assertEqual(data["n_rows"], 4)
            self.assertEqual(len(data["headline"]), 2)
            self.assertGreater(data["headline"][0]["legacy_compat_tok_s"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
