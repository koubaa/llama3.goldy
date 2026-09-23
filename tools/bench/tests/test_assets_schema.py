from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

BENCH = pathlib.Path(__file__).resolve().parents[1]
TOOLS = BENCH.parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from bench import common, schema  # noqa: E402


def _load_fetch():
    path = TOOLS / "fetch_assets.py"
    spec = importlib.util.spec_from_file_location("fetch_assets", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class AssetManifestTests(unittest.TestCase):
    def test_hashes_match_protocol_constants(self):
        fetch = _load_fetch()
        self.assertEqual(fetch.STORIES15M_SHA256, common.CHECKPOINT_SHA256["stories15M.bin"])
        self.assertEqual(fetch.STORIES42M_SHA256, common.CHECKPOINT_SHA256["stories42M.bin"])
        self.assertEqual(fetch.STORIES110M_SHA256, common.CHECKPOINT_SHA256["stories110M.bin"])
        self.assertEqual(fetch.TOKENIZER_SHA256, common.CHECKPOINT_SHA256["tokenizer.bin"])
        self.assertEqual(len(fetch.ASSETS), 4)

    def test_expected_headers(self):
        fetch = _load_fetch()
        self.assertEqual(fetch.ASSETS["stories15M.bin"]["config"]["dim"], 288)
        self.assertEqual(fetch.ASSETS["stories42M.bin"]["config"]["n_layers"], 8)
        self.assertEqual(fetch.ASSETS["stories110M.bin"]["config"]["max_seq_len"], 1024)
        self.assertEqual(fetch.ASSETS["stories15M.bin"]["bytes"], 60_816_028)
        self.assertEqual(fetch.ASSETS["stories42M.bin"]["bytes"], 167_020_572)
        self.assertEqual(fetch.ASSETS["stories110M.bin"]["bytes"], 438_381_596)

    def test_verify_rejects_hash_mismatch(self):
        fetch = _load_fetch()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "tokenizer.bin"
            path.write_bytes(b"not-the-tokenizer")
            with self.assertRaises(SystemExit):
                fetch.verify_file("tokenizer.bin", path)

    def test_sha256_helper(self):
        fetch = _load_fetch()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "x.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                fetch.sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


class SchemaTests(unittest.TestCase):
    def _sample(self, **overrides):
        obj = schema.base_result(
            engine="pytorch-eager",
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
                "prompt": [1, 76, 40],
                "generated": [13, 14],
                "text": common.DREAM_STORY,
                "match_expected": True,
            },
            phases={
                "load_s": 0.1,
                "warmup_s": 0.05,
                "prompt_s": 0.01,
                "ttft_s": 0.012,
                "decode_step_s": [0.001, 0.001],
                "compat_elapsed_s": 0.017,
            },
            metrics={
                "prompt_tok_s": 100.0,
                "decode_tok_s": 1000.0,
                "legacy_compat_tok_s": 2823.0,
            },
        )
        obj.update(overrides)
        return obj

    def test_valid_sample(self):
        schema.validate_result(self._sample())
        line = common.dumps(self._sample())
        schema.validate_result(json.loads(line))

    def test_rejects_unknown_engine(self):
        obj = self._sample()
        obj["engine"] = "mystery"
        with self.assertRaises(schema.SchemaError):
            schema.validate_result(obj)

    def test_rejects_missing_phase(self):
        obj = self._sample()
        del obj["phases"]["ttft_s"]
        with self.assertRaises(schema.SchemaError):
            schema.validate_result(obj)


if __name__ == "__main__":
    unittest.main()
