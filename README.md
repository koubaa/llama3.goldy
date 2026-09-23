# llama3.goldy

FP32 TinyStories generator on [Ammon](../ammon) / [Goldy](https://github.com/koubaa/goldy), replicating [`llama3.cuda`](https://github.com/likejazz/llama3.cuda) commit `424333d1651d2b0fc17d38e9f790e824947e284b`.

This is **not** Meta Llama 3. The first checkpoint is Karpathy’s 15M TinyStories model (Llama-2-style MHA, RoPE θ=10,000) served through `llama3.cuda`’s layout and greedy loop. Ammon’s attention kernel keeps `n_kv_heads` generic so a later GQA checkpoint can exercise grouped-query attention. Llama 3 tokenizer / RoPE scaling / GGUF are out of scope.

There is **no CPU transformer**. Host code loads the checkpoint, tokenizes, tenders a `DecodeStep { token, position }` on the worker deposit, and greedy-argmaxes claimed logits. RMSNorm / GEMV / RoPE / attention / SwiGLU live in Ammon; this crate records the Llama graph and the llama2.c packed-blob layout.

## Assets

```bash
python tools/fetch_assets.py
```

| File | Source | SHA-256 |
|------|--------|---------|
| `models/tokenizer.bin` | [karpathy/llama2.c](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) | `50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361` |
| `models/stories15M.bin` | [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) | `cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a` |
| `models/stories42M.bin` | [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) | `9f65a1000e17d0bc167dd6332e0ce5119a0222a3d920cead5bce413bfab2ee7b` |
| `models/stories110M.bin` | [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) | `515267168726a1ed1317a64a408492e6af3b67c1f71c5bd98c01d9d721803a24` |

All four paths are gitignored. The 42M/110M blobs are for the scaling tier in [BENCHMARKS.md](BENCHMARKS.md); `fetch_assets.py --only required` keeps the 15M + tokenizer pair.

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

- Weights: one retained FP32 tensor blob; this crate maps llama2.c offsets to `CausalAttentionBlockWeights` / `SwiGluBlockWeights`
- Graph ops: Ammon modules (`Embedding`, `CausalAttentionBlock`, `SwiGluBlock`, `RmsNorm`, `Linear`); kernels stay inside Ammon
- `DecodeStep { token, position }`: Ammon control parcel (`DecodeStep::parcel` / `deposit_target`); `MemoryExchange` deposit on the worker root. Tender with `<<` each step; one `worker.submit()`
- KV cache: Ammon `KvCache`; each layer is `[seq_len, n_kv_heads, head_size]` (no `loff`)
- Worker: Ammon records named groups once (`embed`, `layerN/attn`, `layerN/ffn`, `tail`); parcel accesses derive their ordering. Goldy may add a second record if shader specialization promotes; `topology_records == 0`
- Logits: host claim `(&mut submission >> logits).take::<f32>()` after each worker submit; generation samples the `HostView` without another allocation
