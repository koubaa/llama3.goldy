# llama3.goldy

FP32 TinyStories generator on [Ammon](../ammon) / [Goldy](https://github.com/koubaa/goldy), replicating [`llama3.cuda`](https://github.com/likejazz/llama3.cuda) commit `424333d1651d2b0fc17d38e9f790e824947e284b`.

This is **not** Meta Llama 3. The first checkpoint is Karpathy’s 15M TinyStories model (Llama-2-style MHA, RoPE θ=10,000) served through `llama3.cuda`’s layout and greedy loop. Ammon’s attention kernel keeps `n_kv_heads` generic so a later GQA checkpoint can exercise grouped-query attention. Llama 3 tokenizer / RoPE scaling / GGUF are out of scope.

There is **no CPU transformer**. Host code loads the checkpoint, tokenizes, uploads a `DecodeStep { token, position }`, and greedy-argmaxes withdrawn logits. RMSNorm / GEMV / RoPE / attention / SwiGLU live in Ammon; this crate records the Llama graph and the llama2.c packed-blob layout.

## Assets

```bash
python tools/fetch_assets.py
```

| File | Source | SHA-256 |
|------|--------|---------|
| `models/stories15M.bin` | [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) | `cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a` |
| `models/tokenizer.bin` | [karpathy/llama2.c](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) | `50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361` |

Both paths are gitignored.

## Run

CUDA (NVIDIA):

```bash
cargo run --release --features cuda -- "I have a dream"
```

Metal (macOS):

```bash
cargo run --release --features metal -- "I have a dream"
```

Optional flags: `--checkpoint PATH`, `--tokenizer PATH`, `-n 50`.

Expected sample (50 tokens, greedy, llama3.cuda README):

```
I have a dream. He dreams of a big, beautiful garden full of flowers and trees. He dreams of playing with his friends and eating yummy snacks.
One day, he was walking in the garden when he saw
```

Argmax is brittle under FP32 reduction-order differences. If a backend diverges on a later token, compare the prefix and `GOLDY_VALIDATION=api` traces before changing kernels.

## Tests

```bash
cargo test --offline
cargo test --features cuda --test generation -- --nocapture
```

Kernel algebra lives in Ammon:

```bash
cargo test --manifest-path ../ammon/Cargo.toml --features cuda --test kernels
```

On macOS, use `--features metal` in place of `cuda`. Metal hardware is required for the generation gate; a compile-only build is not a substitute.

**Verification (2026-09-19):** CUDA on Windows matched the sample text exactly (`worker records=2` from shader specialization, then retained resubmits). Metal was not run here (no macOS GPU on this machine).

## Goldy mapping

- Weights: one retained FP32 tensor blob; this crate maps llama2.c offsets to `TensorView`s
- Graph ops: Ammon kernels only (`TensorKernels` re-exports Goldy add / semantic matmul)
- `DecodeStep { token, position }`: Ammon control parcel, **separate** upload `Scheme` + `MemoryExchange` deposit so the worker is never mutated
- KV cache: persistent tensors; K/V GEMV writes `loff + pos * kv_dim`
- Worker: unrolled layer graph recorded once (Goldy may add a second record if shader specialization promotes); `topology_records == 0`
- Logits: `bind_withdraw` after each worker submit
