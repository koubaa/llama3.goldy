from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = pathlib.Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters.llama3_cuda.build import SPLICE_MARKER, splice  # noqa: E402
from bench.adapters.llama_cpp.convert import convert_cmd, gguf_path_for  # noqa: E402
from bench.adapters.llama_cpp.run import completion_cmd, parse_perf  # noqa: E402
from bench.adapters.util import REFS_LLAMA3_CUDA, to_wsl_path  # noqa: E402
from bench.schema import SchemaError, validate_result  # noqa: E402


class Llama3CudaAdapterTests(unittest.TestCase):
    def test_ref_contains_splice_marker(self):
        src = REFS_LLAMA3_CUDA / "llama3.cu"
        self.assertTrue(src.exists(), src)
        text = src.read_text(encoding="utf-8", errors="replace")
        self.assertIn(SPLICE_MARKER, text)
        self.assertIn("clock_gettime(CLOCK_REALTIME", text)

    def test_splice_adds_json_and_monotonic_clock(self):
        src = REFS_LLAMA3_CUDA / "llama3.cu"
        with tempfile.TemporaryDirectory() as tmp:
            dest = pathlib.Path(tmp) / "llama3.cu"
            splice(src, dest)
            text = dest.read_text(encoding="utf-8")
            self.assertIn("CLOCK_MONOTONIC", text)
            self.assertIn("schema_version", text)
            self.assertIn("--json", text)
            self.assertIn("cudaDeviceSynchronize", text)
            self.assertNotIn("clock_gettime(CLOCK_REALTIME", text)
            self.assertIn("float *forward(", text)

    def test_committed_patch_applies_to_a_copy(self):
        src = REFS_LLAMA3_CUDA / "llama3.cu"
        patch = (
            pathlib.Path(__file__).resolve().parents[1]
            / "adapters"
            / "llama3_cuda"
            / "llama3.cuda.bench.patch"
        )
        self.assertTrue(patch.exists(), patch)
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            shutil.copy2(src, work / "llama3.cu")
            subprocess.run(
                ["git", "apply", str(patch.resolve())],
                cwd=work,
                check=True,
                capture_output=True,
                text=True,
            )
            text = (work / "llama3.cu").read_text(encoding="utf-8")
            self.assertIn("CLOCK_MONOTONIC", text)
            self.assertIn("schema_version", text)


class LlamaCppAdapterTests(unittest.TestCase):
    def test_convert_cmd_uses_llama2c_tool_and_shared_tokenizer(self):
        ckpt = pathlib.Path("models/stories15M.bin")
        tok = pathlib.Path("models/tokenizer.bin")
        out = gguf_path_for(ckpt)
        cmd = convert_cmd(pathlib.Path("llama-convert-llama2c-to-ggml"), ckpt, tok, out)
        joined = " ".join(cmd)
        self.assertIn("llama-convert-llama2c-to-ggml", joined)
        self.assertIn("--copy-vocab-from-model", joined)
        self.assertIn("tokenizer.bin", joined)
        self.assertTrue(str(out).endswith(".f32.gguf"))

    def test_completion_flags_match_parity_contract(self):
        cmd = completion_cmd(
            pathlib.Path("llama-completion"),
            pathlib.Path("stories15M.f32.gguf"),
            prompt=common.DREAM_PROMPT,
            n_predict=50,
            ctx=256,
            warmup=False,
        )
        joined = " ".join(cmd)
        self.assertIn("-ngl all", joined)
        self.assertIn("-ctk f32", joined)
        self.assertIn("-ctv f32", joined)
        self.assertIn("-fa off", joined)
        self.assertIn("-no-cnv", joined)
        self.assertIn("--no-warmup", joined)
        self.assertIn("--temp 0", joined)

    def test_parse_perf(self):
        text = """
llama_perf_context_print:        load time =     123.00 ms
llama_perf_context_print: prompt eval time =      10.00 ms /     5 tokens (    2.00 ms per token,   500.00 tokens per second)
llama_perf_context_print:        eval time =      32.00 ms /    16 runs   (    2.00 ms per token,   500.00 tokens per second)
"""
        perf = parse_perf(text)
        self.assertAlmostEqual(perf["load_s"], 0.123)
        self.assertAlmostEqual(perf["prompt_s"], 0.01)
        self.assertEqual(perf["prompt_n"], 5)
        self.assertAlmostEqual(perf["eval_s"], 0.032)
        self.assertEqual(perf["eval_n"], 16)


class UtilTests(unittest.TestCase):
    def test_wsl_path(self):
        path = to_wsl_path(r"C:\Dev\kob3\llama3.goldy")
        self.assertTrue(path.startswith("/mnt/c/"), path)
        self.assertNotIn("\\", path)


class SchemaEngineNativeTests(unittest.TestCase):
    def test_llama_cpp_bench_sampling(self):
        obj = {
            "schema_version": 1,
            "engine": "llama.cpp-bench",
            "execution": "native",
            "checkpoint": {
                "path": "x",
                "sha256": "a" * 64,
                "config": {
                    "dim": 1,
                    "hidden_dim": 1,
                    "n_layers": 1,
                    "n_heads": 1,
                    "n_kv_heads": 1,
                    "vocab_size": 1,
                    "max_seq_len": 1,
                },
            },
            "workload": {
                "tier": "scaling",
                "prompt": "",
                "batch": 1,
                "context_len": 8,
                "total_positions": 16,
                "decode_steps": 16,
                "sampling": "engine-native",
            },
            "precision": {"weights": "fp32", "activations": "fp32", "kv": "fp32", "tf32": False},
            "tokens": {"prompt": [], "generated": [], "text": "", "match_expected": True},
            "phases": {
                "load_s": 0,
                "warmup_s": 0,
                "prompt_s": 0,
                "ttft_s": 0,
                "decode_step_s": [],
                "compat_elapsed_s": 0,
            },
            "metrics": {"prompt_tok_s": 0, "decode_tok_s": 0, "legacy_compat_tok_s": 0},
            "engine_native_notes": ["random tokens"],
            "build": {},
            "replay_stats": None,
        }
        validate_result(obj)
        obj["workload"]["sampling"] = "mystery"
        with self.assertRaises(SchemaError):
            validate_result(obj)


if __name__ == "__main__":
    unittest.main()
