from __future__ import annotations

import pathlib
import struct
import sys
import tempfile
import unittest

TOOLS = pathlib.Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench.pytorch.checkpoint import (  # noqa: E402
    HEADER_SIZE,
    Checkpoint,
    Config,
    WeightLayout,
    checkpoint_file_bytes,
)


class LayoutTests(unittest.TestCase):
    def tiny(self) -> Config:
        return Config(4, 8, 1, 2, 2, 4, 4)

    def test_tied_layout_matches_pointer_walk(self):
        cfg = self.tiny()
        layout = WeightLayout.from_config(cfg, True)
        dim, hidden, vocab, n_layers, head_size = 4, 8, 4, 1, 2
        ptr = 0
        self.assertEqual(layout.token_embedding, ptr)
        ptr += vocab * dim
        self.assertEqual(layout.rms_att_weight, ptr)
        ptr += n_layers * dim
        self.assertEqual(layout.wq, ptr)
        ptr += n_layers * dim * dim
        self.assertEqual(layout.wk, ptr)
        ptr += n_layers * dim * dim
        self.assertEqual(layout.wv, ptr)
        ptr += n_layers * dim * dim
        self.assertEqual(layout.wo, ptr)
        ptr += n_layers * dim * dim
        self.assertEqual(layout.rms_ffn_weight, ptr)
        ptr += n_layers * dim
        self.assertEqual(layout.w1, ptr)
        ptr += n_layers * dim * hidden
        self.assertEqual(layout.w2, ptr)
        ptr += n_layers * hidden * dim
        self.assertEqual(layout.w3, ptr)
        ptr += n_layers * dim * hidden
        self.assertEqual(layout.rms_final_weight, ptr)
        ptr += dim
        ptr += 4 * head_size // 2
        ptr += 4 * head_size // 2
        self.assertEqual(layout.wcls, 0)
        self.assertEqual(layout.n_floats, ptr)
        self.assertTrue(layout.shared_classifier)

    def test_layer_offsets_stride(self):
        cfg = Config(4, 8, 2, 2, 2, 4, 4)
        layout = WeightLayout.from_config(cfg, True)
        l0 = layout.layer_offsets(0, cfg)
        l1 = layout.layer_offsets(1, cfg)
        self.assertEqual(l1["rms_att"], l0["rms_att"] + cfg.dim)
        self.assertEqual(l1["wq"], l0["wq"] + cfg.dim * cfg.dim)
        self.assertEqual(l1["wk"], l0["wk"] + cfg.dim * cfg.kv_dim)
        self.assertEqual(l1["w2"], l0["w2"] + cfg.hidden_dim * cfg.dim)

    def test_stories_file_sizes(self):
        cases = [
            (Config(288, 768, 6, 6, 6, 32000, 256), 60_816_028),
            (Config(512, 1376, 8, 8, 8, 32000, 1024), 167_020_572),
            (Config(768, 2048, 12, 12, 12, 32000, 1024), 438_381_596),
        ]
        for cfg, size in cases:
            layout = WeightLayout.from_config(cfg, True)
            self.assertEqual(checkpoint_file_bytes(layout), size, cfg)

    def test_read_untied_and_reject_trunc(self):
        cfg = self.tiny()
        tied = WeightLayout.from_config(cfg, True)
        untied = WeightLayout.from_config(cfg, False)
        self.assertEqual(untied.n_floats, tied.n_floats + cfg.vocab_size * cfg.dim)
        header = struct.pack("<7i", cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, -cfg.vocab_size, cfg.max_seq_len)
        blob = header + bytes(untied.n_floats * 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "untied.bin"
            path.write_bytes(blob)
            ckpt = Checkpoint.read_path(path)
            self.assertEqual(ckpt.config.vocab_size, 4)
            self.assertFalse(ckpt.layout.shared_classifier)
            path.write_bytes(header + bytes(16))
            with self.assertRaises(ValueError):
                Checkpoint.read_path(path)

    def test_header_size(self):
        self.assertEqual(HEADER_SIZE, 28)


if __name__ == "__main__":
    unittest.main()
