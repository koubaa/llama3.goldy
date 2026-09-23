from __future__ import annotations

import pathlib
import sys
import unittest

TOOLS = pathlib.Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench.pytorch.tokenizer import (  # noqa: E402
    Tokenizer,
    apply_dream_prompt_patch,
    printable_piece,
    sample_argmax,
)


class TokenizerTests(unittest.TestCase):
    def toy(self) -> Tokenizer:
        vocab = ["<unk>", "<s>", "</s>", "a", "b", "ab", " "]
        scores = [0.0, 0.0, 0.0, 1.0, 1.0, 10.0, 0.0]
        return Tokenizer.from_parts(vocab, scores, 8)

    def test_encode_bos_dummy_merge(self):
        ids = self.toy().encode("ab", bos=True, eos=False)
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[1], 6)
        self.assertEqual(ids[2], 5)

    def test_decode_strips_space_after_bos(self):
        tok = self.toy()
        self.assertEqual(tok.decode(1, 6), "")
        self.assertEqual(tok.decode(0, 6), " ")

    def test_dream_patch(self):
        ids = [1, 306, 40]
        apply_dream_prompt_patch(ids)
        self.assertEqual(ids[1], 76)
        other = [1, 77]
        apply_dream_prompt_patch(other)
        self.assertEqual(other[1], 77)

    def test_argmax_first_on_ties(self):
        self.assertEqual(sample_argmax([1.0, 3.0, 3.0, 2.0]), 1)

    def test_byte_token_decode(self):
        tok = Tokenizer.from_parts(["<unk>", "<s>", "</s>", "<0x41>"], [0.0] * 4, 8)
        self.assertEqual(tok.decode(0, 3), "A")

    def test_printable_piece_cjk_rewrite(self):
        self.assertEqual(printable_piece(""), "")
        self.assertEqual(printable_piece(bytes([0xC3, 0xA9]).decode("utf-8")), "é")


if __name__ == "__main__":
    unittest.main()
