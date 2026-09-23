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


if __name__ == "__main__":
    unittest.main()
