# Llama benchmark baseline

Informational throughput. Protocol: [`BENCHMARKS.md`](../BENCHMARKS.md).

Goldy rows were refreshed at `20260924T142940Z` after the CUDA GEMV kernel and split-KV flash-decoding attention. Other engines are unchanged from `2026-09-24T01:17:44Z`.

## Machine

- captured: `2026-09-24T14:28:29.738178+00:00`
- host: `WIN-FAQM3SSHSGG`
- os: `Windows 11 10.0.26200`
- gpu: `NVIDIA GeForce RTX 4060 Ti, 591.74, 8188 MiB`
- rustc: `rustc 1.98.0 (88d9e12ae 2026-08-18)`
- torch: `2.14.0+cu130 cuda=13.0 triton=3.8.0 (C:\Dev\kob3\llama3.goldy\tools\bench\.cache\torch-venv\Scripts\python.exe)`
- llama3.cuda pin: `424333d1651d2b0fc17d38e9f790e824947e284b`
- llama.cpp pin: `f072b103714dfa1eee531f80b24512faf38e3dd2`

## Compatibility (stories15M, "I have a dream", 50 positions)

Headline metric is llama3.cuda-style `legacy_compat_tok_s = (pos - 1) / elapsed`.
llama.cpp uses its engine-native tokenizer and is **not** rewritten to match; it has no legacy metric.

| Engine | Exec | Match | Legacy tok/s | Decode tok/s | Prompt tok/s | TTFT s |
|--------|------|-------|--------------|--------------|--------------|--------|
| goldy | native | yes | 1694.7 (1654.8–1711.5) | 1755.0 (1711.2–1776.1) | 1866.3 (1814.1–1897.9) | 0.0027 (0.0026–0.0027) |
| llama.cpp | native | no | — | 2064.8 (2062.4–2098.0) | 1818.2 (1797.3–1828.8) | 0.0027 (0.0027–0.0028) |
| llama3.cuda | native | yes | 1078.9 (1051.9–1091.1) | 1095.0 (1081.4–1108.0) | 1159.8 (987.8–1169.4) | 0.0044 (0.0043–0.0050) |
| pytorch-compile | native | yes | 1318.5 (1270.5–1330.8) | 1459.7 (1389.3–1462.5) | 1449.6 (1368.9–1504.9) | 0.0034 (0.0033–0.0036) |
| pytorch-eager | native | yes | 431.5 (429.3–431.7) | 443.0 (440.8–443.6) | 464.9 (449.6–473.3) | 0.0107 (0.0106–0.0109) |

Execution: goldy=native, llama.cpp-bench=native, llama.cpp=native, llama3.cuda=native, pytorch-compile=native, pytorch-eager=native.

## llama.cpp-bench (engine-native, random tokens)

Do not mix into the compatibility headline.

| Checkpoint | Context | Decode tok/s | Prompt tok/s |
|------------|---------|--------------|--------------|
| stories110M.bin | 8 | 456.9 (456.9–461.5) | — |
| stories110M.bin | 32 | 469.7 (433.8–471.3) | — |
| stories110M.bin | 128 | 471.7 (461.6–474.5) | — |
| stories110M.bin | 224 | 462.3 (456.9–471.6) | — |
| stories15M.bin | 8 | 1939.5 (1739.5–2121.6) | — |
| stories15M.bin | 32 | 1870.2 (1684.2–1929.1) | — |
| stories15M.bin | 128 | 1930.3 (1835.9–2059.3) | — |
| stories15M.bin | 224 | 1757.4 (1660.3–1875.8) | — |
| stories42M.bin | 8 | 943.0 (900.6–956.2) | — |
| stories42M.bin | 32 | 1046.2 (980.9–1054.4) | — |
| stories42M.bin | 128 | 1003.0 (987.1–1003.2) | — |
| stories42M.bin | 224 | 1042.9 (1000.8–1057.0) | — |

## Scaling decode tok/s (median, p10–p90)

### stories110M.bin

| Context | goldy | llama.cpp | llama3.cuda | pytorch-compile | pytorch-eager |
|---------|--------|--------|--------|--------|--------|
| 8 | 448.6 (446.5–457.0) | 464.0 (460.1–467.9) | 414.8 (407.2–432.5) | 468.2 (467.9–470.6) | 259.2 (246.6–260.3) |
| 32 | 452.7 (452.3–456.4) | 468.6 (452.7–470.7) | 418.1 (403.4–424.1) | 466.0 (457.7–468.2) | 189.3 (182.7–192.2) |
| 128 | 450.0 (448.1–451.6) | 469.5 (468.8–478.5) | 422.5 (416.4–422.9) | 457.7 (452.7–462.3) | 245.1 (239.1–252.3) |
| 224 | 442.6 (437.7–446.1) | 464.5 (464.0–467.7) | 379.9 (375.3–387.2) | 406.5 (379.7–446.6) | 233.9 (212.1–247.5) |

TTFT s / prompt tok/s / decode-step median s:

| Context | Engine | TTFT s | Prompt tok/s | Step median s |
|---------|--------|--------|--------------|---------------|
| 8 | goldy | 0.0173 (0.0173–0.0176) | 468.7 (463.2–470.7) | 0.00217 (0.00214–0.00218) |
| 8 | llama.cpp | 0.0177 (0.0176–0.0183) | 453.0 (438.4–455.1) | 0.00216 (0.00214–0.00217) |
| 8 | llama3.cuda | 0.0184 (0.0183–0.0192) | 432.7 (414.6–435.7) | 0.00228 (0.00226–0.00230) |
| 8 | pytorch-compile | 0.0171 (0.0168–0.0171) | 473.4 (466.9–477.8) | 0.00210 (0.00210–0.00210) |
| 8 | pytorch-eager | 0.0313 (0.0306–0.0326) | 254.7 (250.3–262.8) | 0.00386 (0.00375–0.00390) |
| 32 | goldy | 0.0697 (0.0693–0.0700) | 459.4 (457.3–462.1) | 0.00219 (0.00217–0.00219) |
| 32 | llama.cpp | 0.0675 (0.0673–0.0705) | 473.9 (454.0–475.2) | 0.00213 (0.00212–0.00221) |
| 32 | llama3.cuda | 0.0757 (0.0749–0.0786) | 422.2 (406.6–427.2) | 0.00232 (0.00229–0.00232) |
| 32 | pytorch-compile | 0.0704 (0.0697–0.0704) | 453.8 (453.6–458.5) | 0.00209 (0.00207–0.00212) |
| 32 | pytorch-eager | 0.1679 (0.1554–0.1732) | 191.1 (185.7–206.7) | 0.00537 (0.00510–0.00544) |
| 128 | goldy | 0.2807 (0.2806–0.2825) | 456.0 (453.1–456.2) | 0.00220 (0.00218–0.00220) |
| 128 | llama.cpp | 0.2696 (0.2689–0.2697) | 474.8 (474.6–476.0) | 0.00213 (0.00209–0.00213) |
| 128 | llama3.cuda | 0.3043 (0.3040–0.3043) | 420.8 (420.7–421.1) | 0.00234 (0.00233–0.00234) |
| 128 | pytorch-compile | 0.2755 (0.2753–0.2770) | 464.7 (462.2–465.1) | 0.00213 (0.00212–0.00213) |
| 128 | pytorch-eager | 0.5267 (0.5127–0.5298) | 242.9 (241.6–249.6) | 0.00396 (0.00385–0.00401) |
| 224 | goldy | 0.4997 (0.4965–0.5010) | 448.3 (447.1–451.3) | 0.00222 (0.00222–0.00222) |
| 224 | llama.cpp | 0.4741 (0.4738–0.4946) | 472.4 (453.0–472.8) | 0.00215 (0.00214–0.00216) |
| 224 | llama3.cuda | 0.5604 (0.5563–0.5680) | 399.8 (394.9–402.7) | 0.00244 (0.00242–0.00245) |
| 224 | pytorch-compile | 0.5091 (0.4937–0.5153) | 440.2 (434.9–454.0) | 0.00233 (0.00216–0.00241) |
| 224 | pytorch-eager | 0.8978 (0.8918–0.9187) | 249.6 (244.1–251.2) | 0.00395 (0.00391–0.00415) |

### stories15M.bin

| Context | goldy | llama.cpp | llama3.cuda | pytorch-compile | pytorch-eager |
|---------|--------|--------|--------|--------|--------|
| 8 | 1852.8 (1782.9–1854.6) | 2052.0 (2049.7–2091.1) | 1065.1 (1013.6–1098.9) | 1447.4 (1424.5–1457.9) | 446.3 (443.0–458.8) |
| 32 | 1741.5 (1704.1–1775.8) | 2052.0 (2016.8–2117.4) | 1062.9 (1044.3–1097.8) | 1358.5 (1348.5–1440.4) | 438.8 (427.1–440.7) |
| 128 | 1751.0 (1746.6–1786.1) | 2133.7 (2133.7–2145.9) | 1153.3 (1117.3–1159.0) | 1382.3 (1354.7–1457.6) | 418.5 (411.7–419.0) |
| 224 | 1778.9 (1727.9–1797.3) | 1950.6 (1893.9–1990.1) | 1103.1 (1094.1–1121.1) | 1398.8 (1345.5–1449.1) | 418.9 (344.0–446.6) |

TTFT s / prompt tok/s / decode-step median s:

| Context | Engine | TTFT s | Prompt tok/s | Step median s |
|---------|--------|--------|--------------|---------------|
| 8 | goldy | 0.0044 (0.0042–0.0044) | 1831.3 (1811.7–1927.7) | 0.00054 (0.00054–0.00054) |
| 8 | llama.cpp | 0.0042 (0.0042–0.0043) | 1913.9 (1878.1–1921.2) | 0.00049 (0.00048–0.00049) |
| 8 | llama3.cuda | 0.0072 (0.0068–0.0072) | 1110.3 (1107.0–1170.2) | 0.00089 (0.00089–0.00090) |
| 8 | pytorch-compile | 0.0057 (0.0054–0.0057) | 1390.8 (1386.9–1458.2) | 0.00068 (0.00067–0.00068) |
| 8 | pytorch-eager | 0.0175 (0.0174–0.0180) | 452.7 (441.7–457.4) | 0.00209 (0.00208–0.00217) |
| 32 | goldy | 0.0178 (0.0177–0.0179) | 1802.7 (1791.7–1806.6) | 0.00056 (0.00056–0.00056) |
| 32 | llama.cpp | 0.0149 (0.0148–0.0156) | 2144.8 (2052.3–2156.3) | 0.00049 (0.00047–0.00050) |
| 32 | llama3.cuda | 0.0299 (0.0293–0.0304) | 1079.5 (1052.4–1093.5) | 0.00090 (0.00089–0.00090) |
| 32 | pytorch-compile | 0.0244 (0.0225–0.0244) | 1320.6 (1308.6–1426.4) | 0.00066 (0.00066–0.00067) |
| 32 | pytorch-eager | 0.0714 (0.0700–0.0725) | 455.9 (444.6–458.6) | 0.00217 (0.00213–0.00221) |
| 128 | goldy | 0.0713 (0.0713–0.0727) | 1795.4 (1762.5–1796.4) | 0.00055 (0.00054–0.00056) |
| 128 | llama.cpp | 0.0579 (0.0579–0.0579) | 2210.7 (2208.9–2211.6) | 0.00047 (0.00047–0.00047) |
| 128 | llama3.cuda | 0.1127 (0.1116–0.1170) | 1136.3 (1094.2–1148.1) | 0.00085 (0.00084–0.00085) |
| 128 | pytorch-compile | 0.0931 (0.0916–0.0940) | 1376.1 (1362.7–1399.5) | 0.00068 (0.00066–0.00070) |
| 128 | pytorch-eager | 0.3055 (0.2997–0.3058) | 419.0 (418.5–427.0) | 0.00229 (0.00226–0.00229) |
| 224 | goldy | 0.1268 (0.1261–0.1270) | 1767.5 (1764.6–1777.5) | 0.00056 (0.00055–0.00056) |
| 224 | llama.cpp | 0.1063 (0.1040–0.1138) | 2107.6 (1969.8–2153.4) | 0.00051 (0.00050–0.00053) |
| 224 | llama3.cuda | 0.2001 (0.1993–0.2006) | 1120.0 (1116.9–1124.6) | 0.00087 (0.00087–0.00088) |
| 224 | pytorch-compile | 0.1673 (0.1616–0.1703) | 1340.1 (1316.9–1387.6) | 0.00067 (0.00066–0.00067) |
| 224 | pytorch-eager | 0.5229 (0.5193–0.5301) | 428.5 (422.6–431.4) | 0.00233 (0.00217–0.00242) |

### stories42M.bin

| Context | goldy | llama.cpp | llama3.cuda | pytorch-compile | pytorch-eager |
|---------|--------|--------|--------|--------|--------|
| 8 | 964.9 (950.8–967.5) | 967.7 (964.3–1017.7) | 861.2 (851.2–883.1) | 968.6 (936.9–991.5) | 294.2 (273.8–305.2) |
| 32 | 973.0 (950.4–976.3) | 1067.6 (1061.0–1070.1) | 888.6 (850.3–897.2) | 966.7 (870.3–986.9) | 377.0 (374.5–391.3) |
| 128 | 961.3 (961.0–965.4) | 1001.3 (980.5–1058.6) | 875.7 (852.1–892.1) | 904.4 (893.6–975.8) | 363.5 (348.8–371.7) |
| 224 | 957.7 (951.2–958.2) | 1056.3 (1055.7–1062.3) | 899.3 (884.5–901.4) | 970.6 (962.2–974.5) | 363.8 (356.4–370.1) |

TTFT s / prompt tok/s / decode-step median s:

| Context | Engine | TTFT s | Prompt tok/s | Step median s |
|---------|--------|--------|--------------|---------------|
| 8 | goldy | 0.0081 (0.0080–0.0081) | 1011.4 (984.4–1014.3) | 0.00100 (0.00100–0.00100) |
| 8 | llama.cpp | 0.0083 (0.0080–0.0091) | 966.2 (884.4–998.3) | 0.00103 (0.00098–0.00104) |
| 8 | llama3.cuda | 0.0096 (0.0088–0.0099) | 877.4 (869.6–920.6) | 0.00112 (0.00110–0.00112) |
| 8 | pytorch-compile | 0.0081 (0.0080–0.0081) | 984.5 (981.2–997.3) | 0.00101 (0.00100–0.00101) |
| 8 | pytorch-eager | 0.0269 (0.0255–0.0297) | 319.8 (270.0–333.5) | 0.00302 (0.00293–0.00319) |
| 32 | goldy | 0.0321 (0.0321–0.0324) | 996.3 (987.7–996.8) | 0.00101 (0.00101–0.00103) |
| 32 | llama.cpp | 0.0294 (0.0293–0.0295) | 1087.7 (1085.3–1090.4) | 0.00094 (0.00093–0.00094) |
| 32 | llama3.cuda | 0.0346 (0.0345–0.0349) | 924.8 (915.8–926.8) | 0.00110 (0.00110–0.00111) |
| 32 | pytorch-compile | 0.0339 (0.0338–0.0386) | 943.5 (832.1–943.5) | 0.00100 (0.00100–0.00103) |
| 32 | pytorch-eager | 0.0862 (0.0854–0.0874) | 371.1 (364.8–375.7) | 0.00251 (0.00247–0.00256) |
| 128 | goldy | 0.1313 (0.1311–0.1314) | 975.3 (974.9–976.7) | 0.00101 (0.00101–0.00102) |
| 128 | llama.cpp | 0.1247 (0.1228–0.1276) | 1026.5 (1003.6–1042.2) | 0.00100 (0.00095–0.00102) |
| 128 | llama3.cuda | 0.1449 (0.1448–0.1503) | 883.5 (854.7–884.5) | 0.00110 (0.00110–0.00111) |
| 128 | pytorch-compile | 0.1394 (0.1335–0.1399) | 918.3 (915.1–959.9) | 0.00099 (0.00098–0.00101) |
| 128 | pytorch-eager | 0.3634 (0.3504–0.3684) | 352.1 (347.6–365.4) | 0.00263 (0.00263–0.00287) |
| 224 | goldy | 0.2326 (0.2319–0.2331) | 963.1 (961.3–966.3) | 0.00104 (0.00103–0.00104) |
| 224 | llama.cpp | 0.2060 (0.2053–0.2061) | 1087.3 (1086.8–1091.3) | 0.00095 (0.00094–0.00095) |
| 224 | llama3.cuda | 0.2526 (0.2497–0.2549) | 888.0 (879.3–897.6) | 0.00109 (0.00109–0.00110) |
| 224 | pytorch-compile | 0.2400 (0.2320–0.2411) | 933.9 (929.7–966.3) | 0.00101 (0.00099–0.00102) |
| 224 | pytorch-eager | 0.6171 (0.6038–0.6405) | 363.0 (349.9–371.1) | 0.00266 (0.00263–0.00267) |

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
- `goldy`: each step includes worker.submit plus eager host-sink HostView claim
- `llama.cpp`: native llama.cpp Release CUDA build (Ninja + MSVC on Windows); refs/ is unmodified
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
- `llama3.cuda`: native Windows build (nvcc + MSVC) of pinned llama3.cuda with a Win32 POSIX shim (mmap/clock_gettime), not WSL
- `llama3.cuda`: bench_tail.cu splice: CLOCK_MONOTONIC timing, cudaDeviceSynchronize around each measured forward
- `llama3.cuda`: load_s includes tokenizer load and cuBLAS handle creation
- `pytorch-compile`: torch.compile(fullgraph=True, mode=default, automatic dynamic token/pos, no CUDA graphs); compilation happens in warmup
- `goldy`: scaling pads/trims prompt tokens to context_len; filler is last non-BOS id
- `llama.cpp`: scaling prompt pads the text prompt with its last word so it encodes to context_len ids
- `llama.cpp`: --ignore-eos so exactly decode_steps tokens are sampled
- `llama.cpp-bench`: native llama.cpp Release CUDA build (Ninja + MSVC on Windows); refs/ is unmodified
- `llama.cpp-bench`: F32 GGUF via llama-convert-llama2c-to-ggml; GGUF n_ctx_train is 128 and llama.cpp pads the runtime context to 256 cells
- `llama.cpp-bench`: full GPU offload; token_embd stays in a CPU_Mapped buffer (llama.cpp input-layer default, get_rows only)
- `llama.cpp-bench`: llama-bench random tokens; not the compatibility headline
- `llama.cpp-bench`: llama-bench at this commit rejects -ctk/-ctv f32 (accepts f16/bf16/q*), so K/V are f16
- `llama.cpp-bench`: -ngl 99, flash attention off, llama-bench default batch sizes; depth prefill is untimed
- `llama.cpp-bench`: one record per llama-bench sample (samples_ns); decode_step_s is sample_ns/n_gen repeated
- `llama.cpp-bench`: llama-bench test n_prompt=0 n_gen=16 n_depth=8
- `llama3.cuda`: scaling pads/trims prompt tokens to context_len; filler is last non-BOS id
- `pytorch-compile`: scaling pads/trims prompt tokens to context_len; filler is last non-BOS id
- `pytorch-eager`: scaling pads/trims prompt tokens to context_len; filler is last non-BOS id
- `llama.cpp-bench`: llama-bench test n_prompt=0 n_gen=16 n_depth=32
- `llama.cpp-bench`: llama-bench test n_prompt=0 n_gen=16 n_depth=128
- `llama.cpp-bench`: llama-bench test n_prompt=0 n_gen=16 n_depth=224

