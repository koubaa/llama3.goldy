"""llama2.c / llama3.cuda packed FP32 checkpoint (mirrors src/checkpoint.rs)."""

from __future__ import annotations

import array
import pathlib
import struct
from dataclasses import asdict, dataclass


HEADER_FMT = "<7i"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


@dataclass(frozen=True)
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    max_seq_len: int

    def validate(self) -> None:
        if min(
            self.dim,
            self.hidden_dim,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.vocab_size,
            self.max_seq_len,
        ) <= 0:
            raise ValueError(f"checkpoint config has non-positive fields: {self}")
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim {self.dim} is not divisible by n_heads {self.n_heads}")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads {self.n_heads} is not divisible by n_kv_heads {self.n_kv_heads}"
            )

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.dim * self.n_kv_heads // self.n_heads

    @property
    def kv_mul(self) -> int:
        return self.n_heads // self.n_kv_heads

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class WeightLayout:
    token_embedding: int
    rms_att_weight: int
    wq: int
    wk: int
    wv: int
    wo: int
    rms_ffn_weight: int
    w1: int
    w2: int
    w3: int
    rms_final_weight: int
    wcls: int
    n_floats: int
    shared_classifier: bool

    @classmethod
    def from_config(cls, config: Config, shared_classifier: bool) -> WeightLayout:
        config.validate()
        dim = config.dim
        hidden_dim = config.hidden_dim
        n_layers = config.n_layers
        vocab = config.vocab_size
        head_size = config.head_size
        q_dim = config.n_heads * head_size
        kv_dim = config.n_kv_heads * head_size
        ptr = 0
        token_embedding = ptr
        ptr += vocab * dim
        rms_att_weight = ptr
        ptr += n_layers * dim
        wq = ptr
        ptr += n_layers * dim * q_dim
        wk = ptr
        ptr += n_layers * dim * kv_dim
        wv = ptr
        ptr += n_layers * dim * kv_dim
        wo = ptr
        ptr += n_layers * q_dim * dim
        rms_ffn_weight = ptr
        ptr += n_layers * dim
        w1 = ptr
        ptr += n_layers * dim * hidden_dim
        w2 = ptr
        ptr += n_layers * hidden_dim * dim
        w3 = ptr
        ptr += n_layers * dim * hidden_dim
        rms_final_weight = ptr
        ptr += dim
        ptr += config.max_seq_len * head_size // 2
        ptr += config.max_seq_len * head_size // 2
        if shared_classifier:
            wcls = token_embedding
        else:
            wcls = ptr
            ptr += vocab * dim
        return cls(
            token_embedding=token_embedding,
            rms_att_weight=rms_att_weight,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            rms_ffn_weight=rms_ffn_weight,
            w1=w1,
            w2=w2,
            w3=w3,
            rms_final_weight=rms_final_weight,
            wcls=wcls,
            n_floats=ptr,
            shared_classifier=shared_classifier,
        )

    def layer_offsets(self, layer: int, config: Config) -> dict[str, int]:
        dim = config.dim
        hidden = config.hidden_dim
        q = config.n_heads * config.head_size
        kv = config.kv_dim
        l = layer
        return {
            "rms_att": self.rms_att_weight + l * dim,
            "wq": self.wq + l * dim * q,
            "wk": self.wk + l * dim * kv,
            "wv": self.wv + l * dim * kv,
            "wo": self.wo + l * q * dim,
            "rms_ffn": self.rms_ffn_weight + l * dim,
            "w1": self.w1 + l * dim * hidden,
            "w2": self.w2 + l * hidden * dim,
            "w3": self.w3 + l * dim * hidden,
        }


@dataclass
class Checkpoint:
    config: Config
    layout: WeightLayout
    weights: array.array

    @classmethod
    def read_path(cls, path: str | pathlib.Path) -> Checkpoint:
        path = pathlib.Path(path)
        data = path.read_bytes()
        if len(data) < HEADER_SIZE:
            raise ValueError(f"{path}: truncated header")
        raw = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        shared = raw[5] > 0
        config = Config(
            dim=raw[0],
            hidden_dim=raw[1],
            n_layers=raw[2],
            n_heads=raw[3],
            n_kv_heads=raw[4],
            vocab_size=abs(raw[5]),
            max_seq_len=raw[6],
        )
        layout = WeightLayout.from_config(config, shared)
        blob = data[HEADER_SIZE:]
        expected = layout.n_floats * 4
        if len(blob) != expected:
            raise ValueError(
                f"{path}: weight blob is {len(blob)} bytes, expected {expected} "
                f"({layout.n_floats} f32s) for {config}"
            )
        weights = array.array("f")
        weights.frombytes(blob)
        return cls(config=config, layout=layout, weights=weights)

    def floats(self, offset: int, count: int) -> array.array:
        return self.weights[offset : offset + count]


def checkpoint_file_bytes(layout: WeightLayout) -> int:
    return HEADER_SIZE + layout.n_floats * 4
