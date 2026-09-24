from __future__ import annotations

import pathlib
import struct
import sys
import tempfile
import unittest

TOOLS = pathlib.Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench.pytorch.checkpoint import Checkpoint, Config, WeightLayout  # noqa: E402
from bench.pytorch.tokenizer import sample_argmax  # noqa: E402


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_torch(), "torch not installed")
class TinyDecoderTests(unittest.TestCase):
    def _write_ckpt(self, path: pathlib.Path) -> None:
        cfg = Config(4, 8, 1, 2, 2, 4, 4)
        layout = WeightLayout.from_config(cfg, True)
        header = struct.pack(
            "<7i",
            cfg.dim,
            cfg.hidden_dim,
            cfg.n_layers,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.vocab_size,
            cfg.max_seq_len,
        )
        floats = []
        for i in range(layout.n_floats):
            floats.append(((i % 17) - 8) / 16.0)
        blob = header + struct.pack(f"<{layout.n_floats}f", *floats)
        path.write_bytes(blob)

    def test_step_shape_and_greedy_stability(self):
        from bench.pytorch.model import Decoder, configure_precision

        configure_precision("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "tiny.bin"
            self._write_ckpt(path)
            ckpt = Checkpoint.read_path(path)
            dec = Decoder(ckpt, device="cpu")
            import torch

            with torch.inference_mode():
                logits0 = dec.step(1, 0)
                logits1 = dec.step(2, 1)
            self.assertEqual(list(logits0.shape), [4])
            self.assertEqual(list(logits1.shape), [4])
            tok0 = sample_argmax(logits0)
            tok1 = sample_argmax(logits1)
            dec2 = Decoder(ckpt, device="cpu")
            with torch.inference_mode():
                again = sample_argmax(dec2.step(1, 0))
            self.assertEqual(tok0, again)
            self.assertIsInstance(tok1, int)

    def test_eager_and_compile_forwards_agree(self):
        """The in-place/complex-RoPE eager path and the functional compile path are the same math."""
        import torch

        from bench.pytorch.model import Decoder

        cfg = Config(8, 16, 2, 2, 1, 6, 8)  # kv_mul=2 exercises the grouped-query layout
        layout = WeightLayout.from_config(cfg, True)
        header = struct.pack("<7i", 8, 16, 2, 2, 1, 6, 8)
        floats = [((i * 7919) % 23 - 11) / 32.0 for i in range(layout.n_floats)]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "gqa.bin"
            path.write_bytes(header + struct.pack(f"<{layout.n_floats}f", *floats))
            ckpt = Checkpoint.read_path(path)
        a = Decoder(ckpt, device="cpu")
        b = Decoder(ckpt, device="cpu")
        with torch.inference_mode():
            for pos, tok in enumerate([1, 3, 5, 2, 4]):
                la = a._forward_eager(tok, pos)
                lb = b._forward_compiled(tok, pos)
                torch.testing.assert_close(la, lb, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.stack(a.k_cache), torch.stack(b.k_cache), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.stack(a.v_cache), torch.stack(b.v_cache), rtol=1e-5, atol=1e-6)

    def test_tensor_argmax_first_on_ties(self):
        import torch

        self.assertEqual(sample_argmax(torch.tensor([1.0, 3.0, 3.0, 2.0])), 1)

    @unittest.skipIf(_has_torch() and __import__("torch").cuda.is_available(), "CUDA present")
    def test_cuda_request_fails_loudly_without_cuda(self):
        from bench.pytorch.model import configure_precision

        with self.assertRaises(RuntimeError):
            configure_precision("cuda")

    def test_compile_is_optional(self):
        from bench.pytorch.model import Decoder

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "tiny.bin"
            self._write_ckpt(path)
            dec = Decoder(Checkpoint.read_path(path), device="cpu")
            elapsed, err = dec.compile_forward()
            self.assertGreaterEqual(elapsed, 0.0)
            if err is None:
                import torch

                with torch.inference_mode():
                    logits = dec.step(1, 0)
                self.assertEqual(list(logits.shape), [4])


class TorchEnvTests(unittest.TestCase):
    def test_override_and_command(self):
        import os
        from unittest import mock

        from bench.pytorch import env

        with tempfile.TemporaryDirectory() as tmp:
            fake = pathlib.Path(tmp) / "python.exe"
            fake.write_bytes(b"")
            with mock.patch.dict(os.environ, {env.ENV_OVERRIDE: str(fake)}):
                self.assertEqual(env.torch_python(), fake)
                cmd = env.bench_command("--mode", "scaling", compile=True)
            self.assertEqual(cmd[:2], [str(fake), str(env.BENCH_PY)])
            self.assertEqual(cmd[-1], "--compile")
            with mock.patch.dict(os.environ, {env.ENV_OVERRIDE: str(fake) + ".missing"}):
                self.assertIsNone(env.torch_python())
        e = env.bench_env(base={"PATH": ""})
        self.assertIn("TORCHINDUCTOR_CACHE_DIR", e)
        self.assertIn("TRITON_CACHE_DIR", e)


if __name__ == "__main__":
    unittest.main()
