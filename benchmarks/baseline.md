# Llama benchmark baseline

Informational throughput. Protocol: [`BENCHMARKS.md`](../BENCHMARKS.md).

## Machine

- captured: `2026-09-23T04:56:06.231879+00:00`
- host: `WIN-FAQM3SSHSGG`
- os: `Windows 11 10.0.26200`
- gpu: `NVIDIA GeForce RTX 4060 Ti, 591.74, 8188 MiB`
- rustc: `rustc 1.98.0 (88d9e12ae 2026-08-18)`
- torch: `2.14.0+cpu cuda=None`
- llama3.cuda pin: `424333d1651d2b0fc17d38e9f790e824947e284b` (WSL adapter)
- llama.cpp pin: `f072b103714dfa1eee531f80b24512faf38e3dd2` (native adapter)

## Skipped engines

- pytorch-compile: torch.compile bench requires CUDA
- llama3.cuda: WSL bash/nvcc unavailable
- llama.cpp: Visual Studio 2022 MSVC toolset not found
- llama.cpp-bench: Visual Studio 2022 MSVC toolset not found
- pytorch-eager stories42M.bin / stories110M.bin: CPU torch reserved for 15M

## Compatibility (stories15M, "I have a dream", 50 positions)

Headline metric is llama3.cuda-style `legacy_compat_tok_s = (pos - 1) / elapsed`.
llama.cpp is engine-native tokenizer / batched prefill and is **not** rewritten to match.

| Engine | Exec | Match | Legacy tok/s | Decode tok/s | Prompt tok/s | TTFT s |
|--------|------|-------|--------------|--------------|--------------|--------|
| goldy | native | yes | 508.1 (498.9–516.3) | 514.5 (505.7–523.5) | 475.8 (474.2–517.5) | 0.0102 (0.0097–0.0102) |
| pytorch-eager | native | yes | 31.5 (30.8–32.7) | 340.0 (334.7–347.6) | 473.0 (325.5–498.2) | 0.0114 (0.0102–0.0161) |

Execution notes: llama3.cuda is WSL; Goldy / PyTorch / llama.cpp are native.

## Scaling decode tok/s (median, p10–p90)

### stories110M.bin

| Context | goldy | pytorch-eager |
|---------|--------|--------|
| 8 | 122.0 (121.7–122.9) | — |
| 32 | 119.6 (119.4–119.6) | — |
| 128 | 112.9 (108.2–113.1) | — |
| 224 | 106.2 (106.0–107.0) | — |

TTFT s / prompt tok/s / decode-step median s:

| Context | Engine | TTFT s | Prompt tok/s | Step median s |
|---------|--------|--------|--------------|---------------|
| 8 | goldy | 0.0650 (0.0650–0.0653) | 123.7 (122.1–123.8) | 0.00820 (0.00810–0.00820) |
| 32 | goldy | 0.2658 (0.2656–0.2659) | 120.1 (120.0–120.9) | 0.00811 (0.00809–0.00814) |
| 128 | goldy | 1.0777 (1.0776–1.0907) | 118.8 (117.4–118.9) | 0.00897 (0.00888–0.00909) |
| 224 | goldy | 1.9722 (1.9588–1.9736) | 113.6 (113.5–114.4) | 0.00978 (0.00952–0.00979) |

### stories15M.bin

| Context | goldy | pytorch-eager |
|---------|--------|--------|
| 8 | 534.0 (508.3–541.9) | 337.9 (303.6–361.2) |
| 32 | 507.7 (486.4–522.6) | 351.9 (333.5–365.4) |
| 128 | 450.8 (436.5–460.4) | 336.5 (287.6–350.2) |
| 224 | 418.2 (413.9–425.3) | 301.2 (277.1–309.5) |

TTFT s / prompt tok/s / decode-step median s:

| Context | Engine | TTFT s | Prompt tok/s | Step median s |
|---------|--------|--------|--------------|---------------|
| 8 | goldy | 0.0154 (0.0147–0.0157) | 512.3 (506.4–538.1) | 0.00174 (0.00174–0.00183) |
| 8 | pytorch-eager | 0.0222 (0.0184–0.0242) | 375.4 (341.4–435.1) | 0.00263 (0.00255–0.00348) |
| 32 | goldy | 0.0613 (0.0608–0.0631) | 520.8 (506.8–525.8) | 0.00180 (0.00180–0.00188) |
| 32 | pytorch-eager | 0.0793 (0.0649–0.0801) | 401.2 (396.9–498.1) | 0.00260 (0.00247–0.00271) |
| 128 | goldy | 0.2587 (0.2586–0.2590) | 495.1 (494.8–497.2) | 0.00205 (0.00204–0.00206) |
| 128 | pytorch-eager | 0.3390 (0.3358–0.3585) | 377.5 (356.9–380.5) | 0.00280 (0.00264–0.00382) |
| 224 | goldy | 0.4764 (0.4762–0.4771) | 470.5 (470.2–470.6) | 0.00227 (0.00226–0.00228) |
| 224 | pytorch-eager | 0.6572 (0.6079–0.6642) | 340.7 (337.8–370.0) | 0.00299 (0.00274–0.00320) |

### stories42M.bin

| Context | goldy | pytorch-eager |
|---------|--------|--------|
| 8 | 274.6 (271.9–277.0) | — |
| 32 | 272.1 (266.0–272.2) | — |
| 128 | 239.4 (233.9–240.5) | — |
| 224 | 228.1 (225.8–228.4) | — |

TTFT s / prompt tok/s / decode-step median s:

| Context | Engine | TTFT s | Prompt tok/s | Step median s |
|---------|--------|--------|--------------|---------------|
| 8 | goldy | 0.0296 (0.0293–0.0306) | 267.8 (259.2–269.4) | 0.00343 (0.00342–0.00343) |
| 32 | goldy | 0.1180 (0.1172–0.1181) | 270.7 (270.5–272.6) | 0.00349 (0.00349–0.00350) |
| 128 | goldy | 0.4965 (0.4956–0.4967) | 258.1 (257.8–258.3) | 0.00405 (0.00387–0.00407) |
| 224 | goldy | 0.8947 (0.8916–0.9031) | 250.4 (248.1–251.3) | 0.00412 (0.00412–0.00412) |

## Goldy replay stats (last row per group)

| Checkpoint | Tier | Context | records | topology | clean | resubmit_hits |
|------------|------|---------|---------|----------|-------|---------------|
| stories15M.bin | compatibility | 50 | 2 | 0 | 194 | 194 |
| stories110M.bin | scaling | 8 | 2 | 0 | 90 | 90 |
| stories110M.bin | scaling | 32 | 2 | 0 | 186 | 186 |
| stories110M.bin | scaling | 128 | 2 | 0 | 570 | 570 |
| stories110M.bin | scaling | 224 | 2 | 0 | 954 | 954 |
| stories15M.bin | scaling | 8 | 2 | 0 | 90 | 90 |
| stories15M.bin | scaling | 32 | 2 | 0 | 186 | 186 |
| stories15M.bin | scaling | 128 | 2 | 0 | 570 | 570 |
| stories15M.bin | scaling | 224 | 2 | 0 | 954 | 954 |
| stories42M.bin | scaling | 8 | 2 | 0 | 90 | 90 |
| stories42M.bin | scaling | 32 | 2 | 0 | 186 | 186 |
| stories42M.bin | scaling | 128 | 2 | 0 | 570 | 570 |
| stories42M.bin | scaling | 224 | 2 | 0 | 954 | 954 |

## Engine-native notes

- `goldy`: tokenizer load is folded into load_s
- `goldy`: each step includes worker.submit plus full logit HostView claim
- `pytorch-eager`: cuda requested but unavailable; using cpu
- `goldy`: scaling pads/trims prompt tokens to context_len; filler is last non-BOS id
- `pytorch-eager`: scaling pads/trims prompt tokens to context_len; filler is last non-BOS id

