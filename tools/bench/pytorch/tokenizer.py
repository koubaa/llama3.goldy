"""llama2.c BPE tokenizer plus llama3.cuda host compatibility helpers."""

from __future__ import annotations

import pathlib
import struct
from dataclasses import dataclass

BOS_ID = 1
EOS_ID = 2


def apply_dream_prompt_patch(tokens: list[int]) -> None:
    if len(tokens) > 1 and tokens[1] == 306:
        tokens[1] = 76


def printable_piece(piece: str) -> str:
    if not piece:
        return ""
    raw = piece.encode("utf-8", errors="surrogateescape")
    if len(raw) == 1:
        b = raw[0]
        if not (32 <= b < 127 or b in (9, 10, 13, 32) or chr(b).isspace()):
            return ""
    if len(raw) >= 2:
        if raw[0] == 0xC3:
            return chr(raw[1] | 0x40)
        if raw[0] == 0xC2:
            return chr(raw[1])
    return piece


def _parse_byte_token(piece: str) -> int | None:
    if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
        try:
            return int(piece[3:5], 16)
        except ValueError:
            return None
    return None


@dataclass
class Tokenizer:
    vocab: list[str]
    vocab_scores: list[float]
    max_token_length: int
    sorted: list[tuple[str, int]]

    @classmethod
    def from_path(cls, path: str | pathlib.Path, vocab_size: int) -> Tokenizer:
        path = pathlib.Path(path)
        with path.open("rb") as f:
            max_token_length = struct.unpack("<I", f.read(4))[0]
            vocab: list[str] = []
            scores: list[float] = []
            for i in range(vocab_size):
                score = struct.unpack("<f", f.read(4))[0]
                (length,) = struct.unpack("<i", f.read(4))
                if length < 0:
                    raise ValueError(f"tokenizer piece {i} has negative length {length}")
                buf = f.read(length)
                vocab.append(buf.decode("utf-8", errors="surrogateescape"))
                scores.append(score)
        return cls.from_parts(vocab, scores, max_token_length)

    @classmethod
    def from_parts(
        cls, vocab: list[str], vocab_scores: list[float], max_token_length: int
    ) -> Tokenizer:
        sorted_vocab = sorted((s, i) for i, s in enumerate(vocab))
        return cls(vocab, vocab_scores, max_token_length, sorted_vocab)

    def lookup(self, s: str) -> int | None:
        lo, hi = 0, len(self.sorted)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.sorted[mid][0] < s:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(self.sorted) and self.sorted[lo][0] == s:
            return self.sorted[lo][1]
        return None

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        tokens: list[int] = []
        if bos:
            tokens.append(BOS_ID)
        if text:
            dummy = self.lookup(" ")
            if dummy is not None:
                tokens.append(dummy)
        raw = text.encode("utf-8")
        i = 0
        while i < len(raw):
            start = i
            i += 1
            while i < len(raw) and (raw[i] & 0xC0) == 0x80 and i - start < 4:
                i += 1
            piece = raw[start:i]
            try:
                s = piece.decode("utf-8")
            except UnicodeDecodeError:
                s = None
            if s is not None:
                found = self.lookup(s)
                if found is not None:
                    tokens.append(found)
                    continue
            tokens.extend(b + 3 for b in piece)

        while True:
            best_score = -1e10
            best_id = -1
            best_idx = None
            for i in range(len(tokens) - 1):
                merged = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                found = self.lookup(merged)
                if found is not None and self.vocab_scores[found] > best_score:
                    best_score = self.vocab_scores[found]
                    best_id = found
                    best_idx = i
            if best_idx is None:
                break
            tokens[best_idx] = best_id
            del tokens[best_idx + 1]

        if eos:
            tokens.append(EOS_ID)
        return tokens

    def decode(self, prev_token: int, token: int) -> str:
        piece = self.vocab[token] if 0 <= token < len(self.vocab) else ""
        if prev_token == BOS_ID and piece.startswith(" "):
            piece = piece[1:]
        byte_val = _parse_byte_token(piece)
        if byte_val is not None:
            return chr(byte_val)
        return piece


def sample_argmax(logits) -> int:
    """Greedy argmax over host logits; the lowest index wins ties (llama3.cuda `>` scan)."""
    if hasattr(logits, "argmax") and getattr(logits, "ndim", None) == 1:
        # torch/numpy argmax both return the first maximal index.
        return int(logits.argmax())
    best_i = 0
    best = float(logits[0])
    for i, value in enumerate(logits):
        v = float(value)
        if v > best:
            best = v
            best_i = i
    return best_i
