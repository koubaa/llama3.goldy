# Llama benchmark protocol

Checked-in contract for comparing `llama3.goldy`, a direct PyTorch baseline, pinned
`llama3.cuda`, and `llama.cpp` on the Karpathy TinyStories llama2.c checkpoints.
Performance numbers are informational. CI and local tests enforce parsers, asset
hashes, JSON schema, checkpoint layout, and generated-token correctness — not
tokens/s thresholds.

## Result tiers

### Compatibility

The published llama3.cuda README case:

| Field | Value |
|-------|--------|
| Prompt | `"I have a dream"` |
| Total positions | 50 (`while pos < 49` in llama3.cuda) |
| Sampling | Greedy argmax on the full host-side logit vector |
| Tokenizer | llama2.c `tokenizer.bin` plus the llama3.cuda `"I have a dream"` patch (`tokens[1] == 306` → `76`) |
| Expected text | See [`DREAM_STORY`](src/generate.rs) |
| Legacy metric | `(pos - 1) / elapsed`, with `elapsed` started after the first forward (llama3.cuda `time_in_ms`) |

Goldy, PyTorch, and patched llama3.cuda **fail** this mode on text mismatch.
llama.cpp records engine-native tokenizer / BOS / EOS / batched-prefill deviations
instead of silently rewriting the prompt.

### Scaling

Batch 1, greedy host sampling, serial prompt processing (one token per forward):

| Checkpoint | Context lengths | Measured decode steps |
|------------|-----------------|------------------------|
| `stories15M.bin` | 8, 32, 128, 224 | 16 |
| `stories42M.bin` | 8, 32, 128, 224 | 16 |
| `stories110M.bin` | 8, 32, 128, 224 | 16 |

Context is `min(requested, checkpoint.max_seq_len)`. Prompt tokens are truncated
or padded (never with BOS=1) to that length. Decode steps are the greedy tokens
sampled after the prompt is consumed.

## Parity settings

Primary comparison uses:

- FP32 weights, activations, and K/V
- TF32 disabled where the engine exposes a control
- Full GPU residency (weights and KV on device)
- Greedy **host** sampling after a full logit DtoH
- Release / `-O3` builds for timed runs
- Warmups excluded from reported phases
- Repeated runs summarized as **median + p10/p90**

Do not silently “normalize” an engine to this list. Put the difference in
`engine_native_notes` (examples: llama.cpp GGUF metadata `n_ctx=128`, llama.cpp
EOS stop vs llama3.cuda BOS stop, llama-bench random-token decode, WSL vs native).

## Assets

One tokenizer and one llama2.c checkpoint per model, shared by every engine.
Fetch and hash-check with:

```bash
python tools/fetch_assets.py
```

| File | Source | SHA-256 | Size |
|------|--------|---------|------|
| `models/tokenizer.bin` | [karpathy/llama2.c](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) | `50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361` | ≥ 100 000 B |
| `models/stories15M.bin` | [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) | `cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a` | 60 816 028 |
| `models/stories42M.bin` | [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) | `9f65a1000e17d0bc167dd6332e0ce5119a0222a3d920cead5bce413bfab2ee7b` | 167 020 572 |
| `models/stories110M.bin` | [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) | `515267168726a1ed1317a64a408492e6af3b67c1f71c5bd98c01d9d721803a24` | 438 381 596 |

Expected llama2.c headers (positive `vocab_size` ⇒ tied classifier):

| Model | dim | hidden | layers | heads | kv | vocab | seq |
|-------|-----|--------|--------|-------|----|-------|-----|
| 15M | 288 | 768 | 6 | 6 | 6 | 32000 | 256 |
| 42M | 512 | 1376 | 8 | 8 | 8 | 32000 | 1024 |
| 110M | 768 | 2048 | 12 | 12 | 12 | 32000 | 1024 |

## JSON Lines schema

Each repetition is one JSON object on stdout (`schema_version: 1`). Validators
live in [`tools/bench/schema.py`](tools/bench/schema.py). Required fields:

| Key | Meaning |
|-----|---------|
| `schema_version` | `1` |
| `engine` | `goldy` \| `pytorch-eager` \| `pytorch-compile` \| `llama3.cuda` \| `llama.cpp` \| `llama.cpp-bench` |
| `execution` | `native` or `wsl` |
| `checkpoint` | `path`, `sha256`, `config` |
| `workload` | `tier`, `prompt`, `batch`, `context_len`, `total_positions`, `decode_steps`, `sampling` (`greedy`, or `engine-native` for llama-bench) |
| `precision` | `weights`, `activations`, `kv`, `tf32` |
| `tokens` | `prompt`, `generated`, `text`, `match_expected` |
| `phases` | `load_s`, `warmup_s`, `prompt_s`, `ttft_s`, `decode_step_s`, `compat_elapsed_s` |
| `metrics` | `prompt_tok_s`, `decode_tok_s`, `legacy_compat_tok_s` |
| `engine_native_notes` | string list; empty if fully on-contract |
| `build` | compiler/features/commit as available |
| `replay_stats` | Goldy `ReplayStats` or `null` |

### Phase definitions

Timed with a monotonic clock. CUDA engines `synchronize` around each measured
forward. Warmup forwards are not included.

| Phase | Definition |
|-------|------------|
| `load_s` | Checkpoint read, GPU upload, graph/module setup. Tokenizer load may be folded in and must be noted if it is. |
| `warmup_s` | Untimed full pass of the same workload (compile/capture happens here). |
| `prompt_s` | Serial forwards at positions `0 .. n_prompt-2` (forced remaining prompt tokens). |
| `ttft_s` | Wall time from the first measured forward through the forward that yields the first generated token (`prompt_s` + first decode). |
| `decode_step_s` | One entry per sampled token, including the first generated token. Scaling asks for 16. |
| `compat_elapsed_s` | llama3.cuda elapsed: timer starts **after** the first forward; used only for `legacy_compat_tok_s = (pos - 1) / elapsed`. |

Repetitions start at position zero so existing KV rows are overwritten; weights
are not reloaded.

## Engines

| Engine | How | Notes |
|--------|-----|--------|
| Goldy | `src/bin` JSON harness (separate from `llama3-goldy`) | CUDA or Metal; records `ReplayStats`. |
| PyTorch eager | [`tools/bench/pytorch/`](tools/bench/pytorch/) | Primary baseline. `torch.inference_mode()`, FP32, TF32 off. Runs under the CUDA-torch venv `tools/bench/.cache/torch-venv` (override: `KOBA_BENCH_TORCH_PYTHON`); fails rather than falling back to CPU. |
| PyTorch compile | same, `--compile` | Secondary label `pytorch-compile`. Inductor + Triton (`triton-windows` on Windows). Compilation is warmup-only. No CUDA Graphs. |
| llama3.cuda | [`tools/bench/adapters/llama3_cuda/`](tools/bench/adapters/llama3_cuda/) | Copy of commit `424333d1651d2b0fc17d38e9f790e824947e284b`, built in WSL (`KOBA_BENCH_WSL_DISTRO`, default `Ubuntu`) when it has nvcc + g++, otherwise natively with nvcc + MSVC and the Win32 POSIX shim in `win_shim/` (`KOBA_BENCH_LLAMA3_CUDA_EXECUTION=auto\|wsl\|native`). Unpatched binary is verified first; JSON runs use an auditable splice, not edits under `refs/`. |
| llama.cpp | [`tools/bench/adapters/llama_cpp/`](tools/bench/adapters/llama_cpp/) | **Native** Release CUDA build of pinned `refs/llama.cpp` (Ninja inside `vcvars64`, VS 2022 or newer). F32 GGUF via `llama-convert-llama2c-to-ggml`, `-ngl all`, `-ctk f32 -ctv f32`, `-fa off`. |
| llama.cpp-bench | same adapter | `llama-bench` random-token pp/tg. **Never** mixed into the compatibility headline. |

Pinned `refs/llama.cpp` commit: `f072b103714dfa1eee531f80b24512faf38e3dd2`.

## Commands

```bash
python tools/fetch_assets.py
python -m unittest discover -s tools/bench/tests -v

cargo run --release --features cuda --bin llama3-goldy-bench -- --mode compatibility
python tools/bench/adapters/goldy/run.py --mode compatibility

tools/bench/.cache/torch-venv/Scripts/python.exe tools/bench/pytorch/bench.py --mode compatibility
tools/bench/.cache/torch-venv/Scripts/python.exe tools/bench/pytorch/bench.py --mode scaling --checkpoint models/stories15M.bin --context 32

python tools/bench/adapters/llama3_cuda/run.py --mode compatibility
python tools/bench/adapters/llama_cpp/run.py --mode compatibility

# Full matrix (skips engines whose prerequisites are missing)
python tools/bench/run.py --smoke
python tools/bench/run.py
```
