# Llama benchmark baseline

Informational throughput. Protocol: [`BENCHMARKS.md`](../BENCHMARKS.md).

## Machine

- captured: `2026-09-25T19:31:57.558661+00:00`
- host: `MacBookAir.lan`
- os: `Darwin 24.6.0 Darwin Kernel Version 24.6.0: Mon Jan 19 21:58:37 PST 2026; root:xnu-11417.140.69.708.3~1/RELEASE_ARM64_T8103`
- gpu: ``
- rustc: `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- torch: `2.8.0 cuda=None triton=None (/Users/mohamedkoubaa/dev/KOB3/llama3.goldy/tools/bench/.cache/torch-venv/bin/python)`
- llama3.cuda pin: `424333d1651d2b0fc17d38e9f790e824947e284b`
- llama.cpp pin: `f072b103714dfa1eee531f80b24512faf38e3dd2`

## Compatibility (stories15M, "I have a dream", 50 positions)

Headline metric is llama3.cuda-style `legacy_compat_tok_s = (pos - 1) / elapsed`.
llama.cpp uses its engine-native tokenizer and is **not** rewritten to match; it has no legacy metric.

| Engine | Exec | Match | Legacy tok/s | Decode tok/s | Prompt tok/s | TTFT s |
|--------|------|-------|--------------|--------------|--------------|--------|
| goldy | native | yes | 56.3 (56.2–56.3) | 56.6 (56.5–56.6) | 56.2 (55.7–56.7) | 0.0893 (0.0884–0.0894) |
| llama.cpp | native | no | — | 527.1 (517.5–533.0) | 296.6 (271.7–300.9) | 0.0169 (0.0166–0.0184) |
| pytorch-eager | native | yes | 79.6 (79.0–79.8) | 80.2 (79.2–80.3) | 80.5 (78.0–83.0) | 0.0630 (0.0608–0.0632) |

Execution: goldy=native, llama.cpp-bench=native, llama.cpp=native, pytorch-eager=native.

## llama.cpp-bench (engine-native, random tokens)

Do not mix into the compatibility headline.

| Checkpoint | Context | Decode tok/s | Prompt tok/s |
|------------|---------|--------------|--------------|
| stories110M.bin | 8 | 119.9 (119.2–121.0) | — |
| stories110M.bin | 32 | 119.5 (118.4–119.8) | — |
| stories110M.bin | 128 | 120.2 (117.5–120.3) | — |
| stories110M.bin | 224 | 101.4 (99.7–102.0) | — |
| stories15M.bin | 8 | 532.0 (494.3–547.3) | — |
| stories15M.bin | 32 | 536.6 (475.3–541.0) | — |
| stories15M.bin | 128 | 536.3 (533.4–548.6) | — |
| stories15M.bin | 224 | 543.5 (537.6–549.3) | — |
| stories42M.bin | 8 | 250.1 (246.7–258.7) | — |
| stories42M.bin | 32 | 259.7 (251.3–261.7) | — |
| stories42M.bin | 128 | 241.2 (237.4–242.2) | — |
| stories42M.bin | 224 | 275.0 (275.0–275.6) | — |

## Scaling decode tok/s (median, p10–p90)

### stories110M.bin

| Context | goldy | llama.cpp | pytorch-eager |
|---------|--------|--------|--------|
| 8 | 12.8 (12.7–12.8) | 119.0 (118.0–119.1) | 51.8 (51.7–51.9) |
| 32 | 12.8 (12.7–12.8) | 118.9 (118.8–120.0) | 51.5 (51.3–51.7) |
| 128 | 12.7 (12.7–12.7) | 120.9 (118.1–121.1) | 50.6 (50.5–50.9) |
| 224 | 12.7 (12.7–12.8) | 101.6 (101.4–101.7) | 43.7 (43.7–43.7) |

TTFT s / prompt tok/s / decode-step median s:

| Context | Engine | TTFT s | Prompt tok/s | Step median s |
|---------|--------|--------|--------------|---------------|
| 8 | goldy | 0.6260 (0.6256–0.6280) | 12.8 (12.8–12.8) | 0.07824 (0.07809–0.07841) |
| 8 | llama.cpp | 0.0678 (0.0672–0.0684) | 118.1 (116.9–119.0) | 0.00840 (0.00839–0.00847) |
| 8 | pytorch-eager | 0.1568 (0.1553–0.1570) | 51.1 (51.0–51.5) | 0.01925 (0.01917–0.01932) |
| 32 | goldy | 2.5172 (2.5164–2.5173) | 12.7 (12.7–12.7) | 0.07826 (0.07817–0.07847) |
| 32 | llama.cpp | 0.2687 (0.2687–0.2702) | 119.1 (118.4–119.1) | 0.00841 (0.00834–0.00842) |
| 32 | pytorch-eager | 0.6212 (0.6186–0.6216) | 51.5 (51.5–51.7) | 0.01933 (0.01925–0.01943) |
| 128 | goldy | 10.0895 (10.0813–10.1083) | 12.7 (12.7–12.7) | 0.07872 (0.07872–0.07880) |
| 128 | llama.cpp | 1.0699 (1.0672–1.0740) | 119.6 (119.2–119.9) | 0.00827 (0.00826–0.00847) |
| 128 | pytorch-eager | 2.4870 (2.4841–2.4909) | 51.5 (51.4–51.5) | 0.01960 (0.01954–0.01974) |
| 224 | goldy | 17.6460 (17.6280–17.7018) | 12.7 (12.7–12.7) | 0.07886 (0.07824–0.07895) |
| 224 | llama.cpp | 2.2220 (2.2208–2.2629) | 100.8 (99.0–100.9) | 0.00984 (0.00983–0.00986) |
| 224 | pytorch-eager | 5.1629 (5.0447–5.1735) | 43.4 (43.3–44.4) | 0.02294 (0.02286–0.02300) |

### stories15M.bin

| Context | goldy | llama.cpp | pytorch-eager |
|---------|--------|--------|--------|
| 8 | 57.6 (57.3–57.6) | 523.2 (509.0–524.1) | 80.2 (79.9–80.3) |
| 32 | 57.6 (57.5–57.6) | 536.3 (530.2–540.2) | 81.3 (80.6–81.7) |
| 128 | 57.4 (56.7–57.4) | 527.8 (525.1–542.1) | 78.8 (62.2–79.5) |
| 224 | 55.8 (55.7–55.9) | 538.6 (532.8–543.4) | 78.7 (59.9–78.7) |

TTFT s / prompt tok/s / decode-step median s:

| Context | Engine | TTFT s | Prompt tok/s | Step median s |
|---------|--------|--------|--------------|---------------|
| 8 | goldy | 0.1397 (0.1393–0.1398) | 57.3 (57.2–57.5) | 0.01740 (0.01731–0.01740) |
| 8 | llama.cpp | 0.0240 (0.0216–0.0254) | 332.9 (315.3–370.8) | 0.00191 (0.00191–0.00197) |
| 8 | pytorch-eager | 0.0993 (0.0988–0.0998) | 80.8 (80.2–81.2) | 0.01217 (0.01217–0.01219) |
| 32 | goldy | 0.5535 (0.5522–0.5565) | 57.8 (57.5–58.0) | 0.01737 (0.01737–0.01738) |
| 32 | llama.cpp | 0.0671 (0.0641–0.0692) | 477.0 (462.7–499.2) | 0.00186 (0.00185–0.00189) |
| 32 | pytorch-eager | 0.3902 (0.3887–0.3957) | 82.2 (80.9–82.3) | 0.01199 (0.01187–0.01204) |
| 128 | goldy | 2.2206 (2.2097–2.2978) | 57.6 (55.7–57.9) | 0.01743 (0.01743–0.01756) |
| 128 | llama.cpp | 0.2470 (0.2466–0.2472) | 518.2 (517.7–519.0) | 0.00189 (0.00185–0.00190) |
| 128 | pytorch-eager | 1.6282 (1.6220–1.6949) | 78.6 (75.5–79.0) | 0.01243 (0.01227–0.01317) |
| 224 | goldy | 3.9730 (3.9707–3.9749) | 56.4 (56.4–56.4) | 0.01782 (0.01778–0.01785) |
| 224 | llama.cpp | 0.4249 (0.4239–0.4253) | 527.1 (526.7–528.5) | 0.00186 (0.00184–0.00188) |
| 224 | pytorch-eager | 2.8533 (2.8334–3.0278) | 78.5 (74.0–79.1) | 0.01240 (0.01238–0.01252) |

### stories42M.bin

| Context | goldy | llama.cpp | pytorch-eager |
|---------|--------|--------|--------|
| 8 | 26.0 (25.9–26.0) | 256.5 (254.6–257.4) | 60.5 (59.9–60.7) |
| 32 | 25.9 (25.9–25.9) | 259.2 (258.5–260.9) | 60.4 (59.5–60.9) |
| 128 | 26.1 (26.0–26.3) | 240.7 (238.3–240.7) | 59.9 (59.5–60.5) |
| 224 | 24.7 (24.7–24.7) | 273.7 (273.6–275.2) | 60.0 (59.6–60.0) |

TTFT s / prompt tok/s / decode-step median s:

| Context | Engine | TTFT s | Prompt tok/s | Step median s |
|---------|--------|--------|--------------|---------------|
| 8 | goldy | 0.3074 (0.3068–0.3091) | 26.0 (25.9–26.0) | 0.03856 (0.03850–0.03860) |
| 8 | llama.cpp | 0.0368 (0.0337–0.0369) | 217.6 (217.0–237.9) | 0.00390 (0.00389–0.00393) |
| 8 | pytorch-eager | 0.1326 (0.1293–0.1345) | 60.9 (59.6–62.1) | 0.01662 (0.01658–0.01690) |
| 32 | goldy | 1.2292 (1.2274–1.2317) | 26.0 (26.0–26.1) | 0.03846 (0.03842–0.03863) |
| 32 | llama.cpp | 0.1292 (0.1284–0.1300) | 247.6 (246.1–249.2) | 0.00386 (0.00383–0.00387) |
| 32 | pytorch-eager | 0.5363 (0.5298–0.5374) | 59.7 (59.6–60.5) | 0.01666 (0.01663–0.01718) |
| 128 | goldy | 4.9354 (4.9139–4.9573) | 25.9 (25.8–26.1) | 0.03811 (0.03762–0.03839) |
| 128 | llama.cpp | 0.5367 (0.5347–0.5381) | 238.5 (237.9–239.4) | 0.00416 (0.00416–0.00420) |
| 128 | pytorch-eager | 2.1372 (2.1285–2.2043) | 59.9 (58.4–60.1) | 0.01636 (0.01599–0.01644) |
| 224 | goldy | 9.0546 (9.0497–9.0590) | 24.7 (24.7–24.8) | 0.04056 (0.04056–0.04065) |
| 224 | llama.cpp | 0.8204 (0.8193–0.8215) | 273.0 (272.7–273.4) | 0.00365 (0.00363–0.00365) |
| 224 | pytorch-eager | 3.7323 (3.7279–3.7331) | 60.1 (60.0–60.1) | 0.01654 (0.01650–0.01679) |

## Goldy replay stats (last row per group)

| Checkpoint | Tier | Context | records | topology | clean | resubmit_hits |
|------------|------|---------|---------|----------|-------|---------------|
| stories15M.bin | compatibility | 50 | 4 | 0 | 192 |  |
| stories110M.bin | scaling | 8 | 4 | 0 | 88 |  |
| stories110M.bin | scaling | 32 | 4 | 0 | 184 |  |
| stories110M.bin | scaling | 128 | 4 | 0 | 568 |  |
| stories110M.bin | scaling | 224 | 4 | 0 | 952 |  |
| stories15M.bin | scaling | 8 | 4 | 0 | 88 |  |
| stories15M.bin | scaling | 32 | 4 | 0 | 184 |  |
| stories15M.bin | scaling | 128 | 4 | 0 | 568 |  |
| stories15M.bin | scaling | 224 | 4 | 0 | 952 |  |
| stories42M.bin | scaling | 8 | 4 | 0 | 88 |  |
| stories42M.bin | scaling | 32 | 4 | 0 | 184 |  |
| stories42M.bin | scaling | 128 | 4 | 0 | 568 |  |
| stories42M.bin | scaling | 224 | 4 | 0 | 952 |  |

## Engine-native notes

- `goldy`: tokenizer load is folded into load_s
- `goldy`: each step includes worker.submit plus eager host-sink HostView claim
- `goldy`: warmup_s includes 0 extra passes until background compiles settled
- `llama.cpp`: native llama.cpp Release Metal build (Ninja); refs/ is unmodified
- `llama.cpp`: F32 GGUF via llama-convert-llama2c-to-ggml; GGUF n_ctx_train is 128 and llama.cpp pads the runtime context to 256 cells
- `llama.cpp`: full GPU offload; token_embd stays in a CPU_Mapped buffer (llama.cpp input-layer default, get_rows only)
- `llama.cpp`: -ngl all, K/V f32, flash attention off, -b 1 -ub 1 serial prompt forwards
- `llama.cpp`: llama.cpp SPM tokenizer encodes ' I' (306); llama3.cuda patches it to 76, so compatibility text is recorded, not rewritten
- `llama.cpp`: each repetition is a fresh llama-completion process (weights reloaded per repetition)
- `llama.cpp`: load_s is log time from model load start to scheduler reservation (weights + GPU upload + context/KV); excludes process start and CUDA init
- `llama.cpp`: prompt_s is llama.cpp prompt eval over all n_prompt forwards, including the one that yields the first token; ttft_s = prompt_s
- `llama.cpp`: decode_step_s is eval_time/n_eval repeated n_eval times (no per-step clock); n_eval = generated tokens - 1
- `llama.cpp`: tokens.generated is empty: llama-completion does not expose sampled ids
- `llama.cpp`: legacy_compat_tok_s omitted: llama.cpp is not (pos-1)/elapsed
- `llama.cpp`: EOS stops generation (llama3.cuda stops on BOS)
- `llama.cpp`: generated text does not match llama3.cuda DREAM_STORY (engine-native tokenizer)
- `goldy`: scaling pads/trims prompt tokens to context_len; filler is last non-BOS id
- `llama.cpp`: scaling prompt pads the text prompt with its last word so it encodes to context_len ids
- `llama.cpp`: --ignore-eos so exactly decode_steps tokens are sampled
- `llama.cpp-bench`: native llama.cpp Release Metal build (Ninja); refs/ is unmodified
- `llama.cpp-bench`: F32 GGUF via llama-convert-llama2c-to-ggml; GGUF n_ctx_train is 128 and llama.cpp pads the runtime context to 256 cells
- `llama.cpp-bench`: full GPU offload; token_embd stays in a CPU_Mapped buffer (llama.cpp input-layer default, get_rows only)
- `llama.cpp-bench`: llama-bench random tokens; not the compatibility headline
- `llama.cpp-bench`: llama-bench at this commit rejects -ctk/-ctv f32 (accepts f16/bf16/q*), so K/V are f16
- `llama.cpp-bench`: -ngl 99, flash attention off, llama-bench default batch sizes; depth prefill is untimed
- `llama.cpp-bench`: one record per llama-bench sample (samples_ns); decode_step_s is sample_ns/n_gen repeated
- `llama.cpp-bench`: llama-bench test n_prompt=0 n_gen=16 n_depth=8
- `pytorch-eager`: scaling pads/trims prompt tokens to context_len; filler is last non-BOS id
- `llama.cpp-bench`: llama-bench test n_prompt=0 n_gen=16 n_depth=32
- `llama.cpp-bench`: llama-bench test n_prompt=0 n_gen=16 n_depth=128
- `llama.cpp-bench`: llama-bench test n_prompt=0 n_gen=16 n_depth=224

