from __future__ import annotations

import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = pathlib.Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters.llama3_cuda.build import SPLICE_MARKER, splice  # noqa: E402
from bench.adapters.llama_cpp.build import cmake_configure_cmd  # noqa: E402
from bench.adapters.llama_cpp.convert import convert_cmd, gguf_path_for, gguf_tensor_types, verify_f32  # noqa: E402
from bench.adapters.llama_cpp.run import (  # noqa: E402
    bench_cmd,
    completion_cmd,
    parse_gpu_offload,
    parse_perf,
    parse_prompt_tokens,
    require_gpu_offload,
    scaling_prompt,
    wrap_bench_line,
)
from bench.adapters.llama3_cuda import build as llama3_cuda_build  # noqa: E402
from bench.adapters.util import REFS_LLAMA3_CUDA, to_wsl_path, wsl_bash_cmd  # noqa: E402
from bench.schema import SchemaError, validate_result  # noqa: E402


class Llama3CudaAdapterTests(unittest.TestCase):
    def test_ref_contains_splice_marker(self):
        src = REFS_LLAMA3_CUDA / "llama3.cu"
        self.assertTrue(src.exists(), src)
        text = src.read_text(encoding="utf-8", errors="replace")
        self.assertIn(SPLICE_MARKER, text)
        self.assertIn("clock_gettime(CLOCK_REALTIME", text)

    def test_splice_adds_json_and_monotonic_clock(self):
        src = REFS_LLAMA3_CUDA / "llama3.cu"
        with tempfile.TemporaryDirectory() as tmp:
            dest = pathlib.Path(tmp) / "llama3.cu"
            splice(src, dest)
            text = dest.read_text(encoding="utf-8")
            self.assertIn("CLOCK_MONOTONIC", text)
            self.assertIn("schema_version", text)
            self.assertIn("--json", text)
            self.assertIn("cudaDeviceSynchronize", text)
            self.assertNotIn("clock_gettime(CLOCK_REALTIME", text)
            self.assertIn("float *forward(", text)

    def test_committed_patch_applies_to_a_copy(self):
        src = REFS_LLAMA3_CUDA / "llama3.cu"
        patch = (
            pathlib.Path(__file__).resolve().parents[1]
            / "adapters"
            / "llama3_cuda"
            / "llama3.cuda.bench.patch"
        )
        self.assertTrue(patch.exists(), patch)
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            shutil.copy2(src, work / "llama3.cu")
            subprocess.run(
                ["git", "apply", str(patch.resolve())],
                cwd=work,
                check=True,
                capture_output=True,
                text=True,
            )
            text = (work / "llama3.cu").read_text(encoding="utf-8")
            self.assertIn("CLOCK_MONOTONIC", text)
            self.assertIn("schema_version", text)

    def test_splice_reports_platform_execution(self):
        src = REFS_LLAMA3_CUDA / "llama3.cu"
        with tempfile.TemporaryDirectory() as tmp:
            dest = pathlib.Path(tmp) / "llama3.cu"
            splice(src, dest)
            text = dest.read_text(encoding="utf-8")
        self.assertIn('#define BENCH_EXECUTION "native"', text)
        self.assertIn('#define BENCH_EXECUTION "wsl"', text)
        self.assertIn("POSIX shim", text)
        self.assertIn("static BenchRun run;", text)
        self.assertIn("stop_on_bos && next == 1", text)

    def test_native_nvcc_cmd_builds_unmodified_source_with_shim(self):
        out = pathlib.Path("out")
        cmd = llama3_cuda_build.native_nvcc_cmd("unpatched", out)
        self.assertIn("llama3.cu", cmd)
        self.assertIn("-DUSE_CUBLAS=1", cmd)
        self.assertIn("-lcublas", cmd)
        self.assertIn("-std=c++20", cmd)
        self.assertIn("-allow-unsupported-compiler", cmd)
        self.assertIn("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH", cmd)
        self.assertIn(f"-I{llama3_cuda_build.WIN_SHIM}", cmd)
        self.assertEqual(cmd[cmd.index("-include") + 1], str(llama3_cuda_build.CCCL_COMPAT))
        self.assertIn(str(out / "posix_shim.obj"), cmd)
        self.assertIn("-g", cmd)
        self.assertIn("-O3", llama3_cuda_build.native_nvcc_cmd("patched", out))

    def test_wsl_nvcc_never_puts_win_shim_on_include_path(self):
        script = llama3_cuda_build.wsl_nvcc_script("patched", pathlib.Path(r"C:\x\y"))
        self.assertIn("cd '/mnt/c/x/y'", script)
        self.assertIn("-DUSE_CUBLAS=1", script)
        self.assertIn("-lcublas", script)
        self.assertIn("cccl_compat.h", script)
        self.assertNotIn("-I", script)

    def test_win_shim_covers_llama3_cu_posix_includes(self):
        shim = llama3_cuda_build.WIN_SHIM
        self.assertTrue((shim / "unistd.h").exists())
        self.assertTrue((shim / "sys" / "mman.h").exists())
        impl = (shim / "posix_shim.c").read_text(encoding="utf-8")
        for sym in ("MapViewOfFile", "UnmapViewOfFile", "QueryPerformanceCounter"):
            self.assertIn(sym, impl)
        header = (shim / "posix_shim.h").read_text(encoding="utf-8")
        self.assertNotIn("#include <windows.h>", header)


LLAMA_CPP_COMPLETION_LOG = """\
0.00.015.050 I llama_completion: load the model and apply lora adapter, if any
0.00.094.231 I llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 4060 Ti) (0000:01:00.0) - 7063 MiB free
0.00.110.181 I load_tensors: offloaded 7/7 layers to GPU
0.00.110.183 I load_tensors:   CPU_Mapped model buffer size =    35.16 MiB
0.00.110.184 I load_tensors:        CUDA0 model buffer size =    57.95 MiB
0.00.129.073 I llama_context: flash_attn            = disabled
0.00.129.560 I llama_kv_cache:      CUDA0 KV buffer size =     3.38 MiB
0.00.129.746 I llama_kv_cache: size =    3.38 MiB (   256 cells,   6 layers,  1/1 seqs), K (f32):    1.69 MiB, V (f32):    1.69 MiB
0.00.130.617 I sched_reserve: reserve took 0.86 ms, sched copies = 1
0.00.130.775 I llama_completion: number of tokens in prompt = 5
0.00.130.776 I      1 -> '<s>'
0.00.130.776 I    306 -> ' I'
0.00.130.776 I    505 -> ' have'
0.00.130.776 I    263 -> ' a'
0.00.130.777 I  12561 -> ' dream'
0.00.130.793 I sampler chain: logits -> ?penalties -> top-k -> dist
0.00.163.640 I common_perf_print:        load time =      21.04 ms
0.00.163.641 I common_perf_print: prompt eval time =      20.94 ms /     5 tokens (    4.19 ms per token,   238.77 tokens per second)
0.00.163.642 I common_perf_print:        eval time =      21.49 ms /    44 runs   (    0.49 ms per token,  2047.27 tokens per second)
0.00.163.642 I common_perf_print:       total time =      43.92 ms /    49 tokens
"""


class LlamaCppAdapterTests(unittest.TestCase):
    def test_cmake_uses_ninja_and_msvc_cuda_workarounds(self):
        env = {"LLAMA_CPP_CMAKE": "cmake", "LLAMA_CPP_NINJA": "ninja"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("LLAMA_CPP_CUDA_ARCH", None)
            cmd = cmake_configure_cmd()
        joined = " ".join(cmd)
        self.assertIn("-G Ninja", joined)
        self.assertNotIn("Visual Studio", joined)
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", cmd)
        self.assertIn("-DGGML_CUDA=ON", cmd)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=89", cmd)
        self.assertIn("-DLLAMA_BUILD_SERVER=OFF", cmd)
        if os.name == "nt":
            self.assertIn("-DCMAKE_CXX_COMPILER=cl", cmd)
            self.assertIn("-allow-unsupported-compiler", joined)
            self.assertIn("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH", joined)

    def test_cuda_arch_env_override(self):
        env = {"LLAMA_CPP_CMAKE": "cmake", "LLAMA_CPP_NINJA": "ninja", "LLAMA_CPP_CUDA_ARCH": "86;89"}
        with mock.patch.dict(os.environ, env):
            self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86;89", cmake_configure_cmd())

    def test_convert_cmd_uses_llama2c_tool_and_shared_tokenizer(self):
        ckpt = pathlib.Path("models/stories15M.bin")
        tok = pathlib.Path("models/tokenizer.bin")
        out = gguf_path_for(ckpt)
        cmd = convert_cmd(pathlib.Path("llama-convert-llama2c-to-ggml"), ckpt, tok, out)
        joined = " ".join(cmd)
        self.assertIn("llama-convert-llama2c-to-ggml", joined)
        self.assertIn("--copy-vocab-from-model", joined)
        self.assertIn("tokenizer.bin", joined)
        self.assertTrue(str(out).endswith(".f32.gguf"))

    def test_completion_flags_match_parity_contract(self):
        cmd = completion_cmd(
            pathlib.Path("llama-completion"),
            pathlib.Path("stories15M.f32.gguf"),
            prompt=common.DREAM_PROMPT,
            n_predict=50,
            ctx=256,
            warmup=False,
        )
        joined = " ".join(cmd)
        self.assertIn("-ngl all", joined)
        self.assertIn("-fit off", joined)
        self.assertIn("-ctk f32", joined)
        self.assertIn("-ctv f32", joined)
        self.assertIn("-fa off", joined)
        self.assertIn("-no-cnv", joined)
        self.assertIn("--no-warmup", joined)
        self.assertIn("--temp 0", joined)
        self.assertIn("--verbose-prompt", joined)
        self.assertIn("-lv 4", joined)
        self.assertNotIn("--ignore-eos", joined)

    def test_bench_cmd_uses_f16_kv_and_verbose(self):
        cmd = bench_cmd(pathlib.Path("llama-bench"), pathlib.Path("m.gguf"), n_prompt=0, n_gen=16, depth=32, reps=3)
        joined = " ".join(cmd)
        self.assertIn("-ctk f16 -ctv f16", joined)
        self.assertIn("-ngl 99", joined)
        self.assertIn("-fa off", joined)
        self.assertIn("-d 32", joined)
        self.assertIn("-o jsonl", joined)
        self.assertIn("-v", cmd)

    def test_parse_real_completion_log(self):
        perf = parse_perf(LLAMA_CPP_COMPLETION_LOG)
        self.assertAlmostEqual(perf["load_s"], 0.130617 - 0.015050, places=6)
        self.assertAlmostEqual(perf["prompt_s"], 0.02094)
        self.assertEqual(perf["prompt_n"], 5)
        self.assertAlmostEqual(perf["eval_s"], 0.02149)
        self.assertEqual(perf["eval_n"], 44)
        self.assertEqual(parse_prompt_tokens(LLAMA_CPP_COMPLETION_LOG), [1, 306, 505, 263, 12561])

    def test_gpu_offload_detection(self):
        info = parse_gpu_offload(LLAMA_CPP_COMPLETION_LOG)
        self.assertEqual(info["device"], "CUDA0")
        self.assertEqual((info["offloaded_layers"], info["total_layers"]), (7, 7))
        self.assertEqual(info["kv_types"], ["f32", "f32"])
        self.assertEqual(info["flash_attn"], "disabled")
        require_gpu_offload(info, kv_type="f32")
        with self.assertRaises(SystemExit):
            require_gpu_offload(info, kv_type="f16")
        cpu_log = LLAMA_CPP_COMPLETION_LOG.replace("offloaded 7/7", "offloaded 0/7").replace("CUDA0 KV", "CPU KV")
        with self.assertRaises(SystemExit):
            require_gpu_offload(parse_gpu_offload(cpu_log), kv_type="f32")

    def test_scaling_prompt_pads_with_last_token(self):
        vocab = {"I": 306, "have": 505, "a": 263, "dream": 12561}

        def fake_tokenize(text: str) -> list[int]:
            return [1] + [vocab[w] for w in text.split()]

        base = fake_tokenize("I have a dream")
        text, ids = scaling_prompt("I have a dream", base, 8, fake_tokenize)
        self.assertEqual(ids, [1, 306, 505, 263, 12561, 12561, 12561, 12561])
        self.assertEqual(text, "I have a dream dream dream dream")
        with self.assertRaises(SystemExit):
            scaling_prompt("I have a dream", base, 4, fake_tokenize)

    def test_gguf_tensor_types_reads_header(self):
        def gguf_str(s: str) -> bytes:
            b = s.encode()
            return struct.pack("<Q", len(b)) + b

        header = b"GGUF" + struct.pack("<IQQ", 3, 2, 2)
        header += gguf_str("general.name") + struct.pack("<I", 8) + gguf_str("tiny")
        header += gguf_str("tokenizer.ggml.scores") + struct.pack("<IIQ", 9, 6, 3) + struct.pack("<3f", 0, 1, 2)
        header += gguf_str("a.weight") + struct.pack("<I2QIQ", 2, 4, 4, 0, 0)
        header += gguf_str("b.weight") + struct.pack("<I1QIQ", 1, 4, 1, 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "t.gguf"
            path.write_bytes(header)
            self.assertEqual(gguf_tensor_types(path), {"a.weight": 0, "b.weight": 1})
            with self.assertRaises(SystemExit):
                verify_f32(path)

    def test_wrap_bench_line_emits_one_valid_record_per_sample(self):
        raw = {"n_prompt": 0, "n_gen": 16, "n_depth": 32, "avg_ns": 8e6, "avg_ts": 2000.0, "samples_ns": [8e6, 1.6e7]}
        config = {"dim": 288, "hidden_dim": 768, "n_layers": 6, "n_heads": 6,
                  "n_kv_heads": 6, "vocab_size": 32000, "max_seq_len": 256}
        objs = [
            wrap_bench_line(raw, checkpoint=pathlib.Path("x.bin"), sha="a" * 64, config=config,
                            gguf=pathlib.Path("x.gguf"), sample_ns=ns)
            for ns in raw["samples_ns"]
        ]
        for obj in objs:
            validate_result(obj)
            self.assertEqual(obj["precision"]["kv"], "fp16")
            self.assertEqual(obj["workload"]["context_len"], 32)
            self.assertNotIn("samples_ns", obj["build"]["llama_bench"])
        self.assertAlmostEqual(objs[0]["metrics"]["decode_tok_s"], 2000.0)
        self.assertAlmostEqual(objs[1]["metrics"]["decode_tok_s"], 1000.0)

    def test_parse_perf(self):
        text = """
llama_perf_context_print:        load time =     123.00 ms
llama_perf_context_print: prompt eval time =      10.00 ms /     5 tokens (    2.00 ms per token,   500.00 tokens per second)
llama_perf_context_print:        eval time =      32.00 ms /    16 runs   (    2.00 ms per token,   500.00 tokens per second)
"""
        perf = parse_perf(text)
        self.assertAlmostEqual(perf["load_s"], 0.123)
        self.assertAlmostEqual(perf["prompt_s"], 0.01)
        self.assertEqual(perf["prompt_n"], 5)
        self.assertAlmostEqual(perf["eval_s"], 0.032)
        self.assertEqual(perf["eval_n"], 16)


class UtilTests(unittest.TestCase):
    def test_wsl_path(self):
        path = to_wsl_path(r"C:\Dev\kob3\llama3.goldy")
        self.assertTrue(path.startswith("/mnt/c/"), path)
        self.assertNotIn("\\", path)

    def test_wsl_bash_cmd_targets_named_distro(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KOBA_BENCH_WSL_DISTRO", None)
            cmd = wsl_bash_cmd("true")
        self.assertEqual(cmd[:6], ["wsl", "-d", "Ubuntu", "-e", "bash", "-lc"])
        self.assertIn("/usr/local/cuda/bin", cmd[-1])
        self.assertTrue(cmd[-1].endswith("true"))
        with mock.patch.dict(os.environ, {"KOBA_BENCH_WSL_DISTRO": "Ubuntu-24.04"}):
            self.assertEqual(wsl_bash_cmd("true")[2], "Ubuntu-24.04")


class SchemaEngineNativeTests(unittest.TestCase):
    def test_llama_cpp_bench_sampling(self):
        obj = {
            "schema_version": 1,
            "engine": "llama.cpp-bench",
            "execution": "native",
            "checkpoint": {
                "path": "x",
                "sha256": "a" * 64,
                "config": {
                    "dim": 1,
                    "hidden_dim": 1,
                    "n_layers": 1,
                    "n_heads": 1,
                    "n_kv_heads": 1,
                    "vocab_size": 1,
                    "max_seq_len": 1,
                },
            },
            "workload": {
                "tier": "scaling",
                "prompt": "",
                "batch": 1,
                "context_len": 8,
                "total_positions": 16,
                "decode_steps": 16,
                "sampling": "engine-native",
            },
            "precision": {"weights": "fp32", "activations": "fp32", "kv": "fp32", "tf32": False},
            "tokens": {"prompt": [], "generated": [], "text": "", "match_expected": True},
            "phases": {
                "load_s": 0,
                "warmup_s": 0,
                "prompt_s": 0,
                "ttft_s": 0,
                "decode_step_s": [],
                "compat_elapsed_s": 0,
            },
            "metrics": {"prompt_tok_s": 0, "decode_tok_s": 0, "legacy_compat_tok_s": 0},
            "engine_native_notes": ["random tokens"],
            "build": {},
            "replay_stats": None,
        }
        validate_result(obj)
        obj["workload"]["sampling"] = "mystery"
        with self.assertRaises(SchemaError):
            validate_result(obj)


if __name__ == "__main__":
    unittest.main()
