//! Phased JSON Lines benchmark for the retained Goldy decoder.
//!
//! Does not change the user-facing `llama3-goldy` timing. Each repetition starts
//! at position zero so existing KV rows are overwritten; weights stay loaded.

fn main() -> anyhow::Result<()> {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        anyhow::bail!("build with --features cuda and/or --features metal");
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        run()
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn run() -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    use llama3_goldy::bench::{
        expected_checkpoint_sha256, pad_or_trim, sha256_file, COMPAT_TOTAL_POSITIONS, DREAM_PROMPT,
        SCALING_DECODE_STEPS,
    };
    use llama3_goldy::checkpoint::Checkpoint;
    use llama3_goldy::model::Model;
    use llama3_goldy::tokenizer::{apply_dream_prompt_patch, Tokenizer};
    use llama3_goldy::{LLAMA3_CUDA_COMMIT, LLAMA_CPP_COMMIT};
    use ammon::Tokenizer as _;
    use serde_json::json;
    use std::path::PathBuf;
    use std::time::Instant;

    fn print_usage() {
        eprintln!(
            "Usage: llama3-goldy-bench [--mode compatibility|scaling] [--checkpoint PATH]\n\
             [--tokenizer PATH] [--prompt TEXT] [--context N] [--decode-steps N]\n\
             [-n TOTAL_POSITIONS] [--warmups N] [--reps N] [--no-dream-patch]"
        );
    }

    let mut checkpoint = PathBuf::from("models/stories15M.bin");
    let mut tokenizer_path = PathBuf::from("models/tokenizer.bin");
    let mut prompt = DREAM_PROMPT.to_string();
    let mut mode = "compatibility".to_string();
    let mut n_tokens: u32 = COMPAT_TOTAL_POSITIONS;
    let mut context: Option<u32> = None;
    let mut decode_steps: u32 = SCALING_DECODE_STEPS;
    let mut warmups: u32 = 1;
    let mut reps: u32 = 1;
    let mut dream_patch = true;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            "--checkpoint" | "-c" => {
                checkpoint = PathBuf::from(args.next().context("missing --checkpoint path")?);
            }
            "--tokenizer" | "-z" => {
                tokenizer_path = PathBuf::from(args.next().context("missing --tokenizer path")?);
            }
            "--mode" => {
                mode = args.next().context("missing --mode")?;
            }
            "--prompt" => {
                prompt = args.next().context("missing --prompt")?;
            }
            "--context" => {
                context = Some(args.next().context("missing --context")?.parse()?);
            }
            "--decode-steps" => {
                decode_steps = args.next().context("missing --decode-steps")?.parse()?;
            }
            "-n" | "--total-positions" | "--n" => {
                n_tokens = args.next().context("missing -n value")?.parse()?;
            }
            "--warmups" => {
                warmups = args.next().context("missing --warmups")?.parse()?;
            }
            "--reps" => {
                reps = args.next().context("missing --reps")?.parse()?;
            }
            "--no-dream-patch" => dream_patch = false,
            flag if flag.starts_with('-') => bail!("unknown flag {flag}"),
            other => prompt = other.to_string(),
        }
    }
    if mode != "compatibility" && mode != "scaling" {
        bail!("--mode must be compatibility or scaling");
    }

    let sha = sha256_file(&checkpoint)?;
    if let Some(expected) = checkpoint
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(expected_checkpoint_sha256)
    {
        if sha != expected {
            bail!("checkpoint hash {sha} != pinned {expected}");
        }
    }

    let load_start = Instant::now();
    let ckpt = Checkpoint::read_path(&checkpoint)?;
    let tokenizer = Tokenizer::from_path(&tokenizer_path, ckpt.config.vocab_size())?;
    let mut model = Model::load(&ckpt)?;
    let load_s = load_start.elapsed().as_secs_f64();

    let mut prompt_tokens = tokenizer.encode(&prompt, true, false);
    if dream_patch {
        apply_dream_prompt_patch(&mut prompt_tokens);
    }
    let mut notes: Vec<String> = vec![
        "tokenizer load is folded into load_s".into(),
        "each step includes worker.submit plus eager host-sink HostView claim".into(),
    ];
    if std::env::var_os("GOLDY_VALIDATION").is_some() {
        notes.push("GOLDY_VALIDATION is set; timed submits may be synchronous".into());
    }
    if std::env::var_os("GOLDY_GPU_PROFILE").is_some() {
        notes.push("GOLDY_GPU_PROFILE is set; command-buffer reuse may be disabled".into());
    }

    let (context_len, max_new, reported_decode_steps) = if mode == "compatibility" {
        let max_new = n_tokens.min(ckpt.config.max_seq_len() as u32);
        let decode = max_new.saturating_sub(prompt_tokens.len() as u32);
        (max_new, max_new, decode)
    } else {
        let context_len = context
            .unwrap_or(llama3_goldy::bench::SCALING_CONTEXT_LENGTHS[0])
            .min(ckpt.config.max_seq_len() as u32);
        prompt_tokens = pad_or_trim(&prompt_tokens, context_len as usize)?;
        let max_new = (context_len + decode_steps).min(ckpt.config.max_seq_len() as u32 + 1);
        notes.push("scaling pads/trims prompt tokens to context_len; filler is last non-BOS id".into());
        (context_len, max_new, decode_steps)
    };

    let stop_on_bos = mode == "compatibility";
    let mut warmup_s = 0.0;
    for _ in 0..warmups {
        let t0 = Instant::now();
        let _ = generate_loop(&mut model, &prompt_tokens, max_new, stop_on_bos)?;
        warmup_s += t0.elapsed().as_secs_f64();
    }

    let features = {
        let mut f = Vec::new();
        if cfg!(feature = "cuda") {
            f.push("cuda");
        }
        if cfg!(feature = "metal") {
            f.push("metal");
        }
        f
    };
    let build = json!({
        "crate": "llama3-goldy",
        "bin": "llama3-goldy-bench",
        "features": features,
        "debug_assertions": cfg!(debug_assertions),
        "llama3_cuda_commit": LLAMA3_CUDA_COMMIT,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
    });
    let checkpoint_obj = json!({
        "path": checkpoint.display().to_string(),
        "sha256": sha,
        "config": {
            "dim": ckpt.config.dim,
            "hidden_dim": ckpt.config.hidden_dim,
            "n_layers": ckpt.config.n_layers,
            "n_heads": ckpt.config.n_heads,
            "n_kv_heads": ckpt.config.n_kv_heads,
            "vocab_size": ckpt.config.vocab_size,
            "max_seq_len": ckpt.config.max_seq_len,
        }
    });
    let workload = json!({
        "tier": mode,
        "prompt": prompt,
        "batch": 1,
        "context_len": context_len,
        "total_positions": max_new,
        "decode_steps": reported_decode_steps,
        "sampling": "greedy",
    });

    for _ in 0..reps.max(1) {
        let out = generate_loop(&mut model, &prompt_tokens, max_new, stop_on_bos)?;
        emit_result(
            &mut model,
            &tokenizer,
            &prompt_tokens,
            out,
            &mode,
            load_s,
            warmup_s,
            &notes,
            &build,
            &checkpoint_obj,
            &workload,
        )?;
    }
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn emit_result(
    model: &mut llama3_goldy::model::Model,
    tokenizer: &llama3_goldy::tokenizer::Tokenizer,
    prompt_tokens: &[i32],
    out: LoopOut,
    mode: &str,
    load_s: f64,
    warmup_s: f64,
    notes: &[String],
    build: &serde_json::Value,
    checkpoint_obj: &serde_json::Value,
    workload: &serde_json::Value,
) -> anyhow::Result<()> {
    use anyhow::bail;
    use llama3_goldy::bench::{rates, DREAM_STORY, SCHEMA_VERSION};
    use serde_json::{json, Value};

    let text = decode_text(tokenizer, prompt_tokens, &out.generated);
    let match_expected = if mode == "compatibility" {
        text == DREAM_STORY
    } else {
        true
    };
    if mode == "compatibility" && !match_expected {
        bail!("token mismatch\n--- got ---\n{text}\n--- expected ---\n{DREAM_STORY}");
    }
    let decode_s: f64 = out.decode_step_s.iter().copied().sum();
    let (prompt_tok_s, decode_tok_s, legacy) = rates(
        prompt_tokens.len(),
        out.prompt_s,
        out.decode_step_s.len(),
        decode_s,
        out.pos,
        out.compat_elapsed_s,
    );
    let fusion = model.fusion_report();
    if !fusion.regions.is_empty() || !fusion.rejected.is_empty() {
        let fused: usize = fusion.regions.iter().map(|r| r.nodes.len()).sum();
        eprintln!(
            "fusion: {} regions of {fused} nodes, {} nodes executed",
            fusion.regions.len(),
            model.executed_node_count()
        );
        for r in &fusion.regions {
            let labels: Vec<String> = r.labels.iter().map(ToString::to_string).collect();
            eprintln!("  {:?} {:?} {:?} {}", r.status, r.schedule, r.cost, labels.join(" "));
        }
        for r in &fusion.rejected {
            let labels: Vec<String> = r.labels.iter().map(ToString::to_string).collect();
            eprintln!("  rejected {}: {}", labels.join(" "), r.reason);
        }
    }
    let stats = model.replay_stats();
    let mut replay = serde_json::Map::new();
    replay.insert("records".into(), json!(stats.records));
    replay.insert("topology_records".into(), json!(stats.topology_records));
    replay.insert("clean_submits".into(), json!(stats.clean_submits));
    replay.insert("specialization_warms".into(), json!(stats.specialization_warms));
    replay.insert(
        "specialization_promotions".into(),
        json!(stats.specialization_promotions),
    );
    replay.insert(
        "specialization_demotions".into(),
        json!(stats.specialization_demotions),
    );
    #[cfg(not(feature = "metal"))]
    replay.insert("resubmit_hits".into(), json!(stats.resubmit_hits));

    let obj = json!({
        "schema_version": SCHEMA_VERSION,
        "engine": "goldy",
        "execution": "native",
        "checkpoint": checkpoint_obj,
        "workload": workload,
        "precision": {
            "weights": "fp32",
            "activations": "fp32",
            "kv": "fp32",
            "tf32": false,
        },
        "tokens": {
            "prompt": prompt_tokens,
            "generated": out.generated,
            "text": text,
            "match_expected": match_expected,
        },
        "phases": {
            "load_s": load_s,
            "warmup_s": warmup_s,
            "prompt_s": out.prompt_s,
            "ttft_s": out.ttft_s,
            "decode_step_s": out.decode_step_s,
            "compat_elapsed_s": out.compat_elapsed_s,
        },
        "metrics": {
            "prompt_tok_s": prompt_tok_s,
            "decode_tok_s": decode_tok_s,
            "legacy_compat_tok_s": legacy,
        },
        "engine_native_notes": notes,
        "build": build,
        "replay_stats": Value::Object(replay),
    });
    println!("{}", serde_json::to_string(&obj)?);
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
struct LoopOut {
    generated: Vec<i32>,
    pos: u32,
    prompt_s: f64,
    ttft_s: f64,
    decode_step_s: Vec<f64>,
    compat_elapsed_s: f64,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn generate_loop(
    model: &mut llama3_goldy::model::Model,
    prompt_tokens: &[i32],
    max_new_tokens: u32,
    stop_on_bos: bool,
) -> anyhow::Result<LoopOut> {
    use ammon::sample_argmax;
    use llama3_goldy::bench::BOS_ID;
    use std::time::Instant;

    anyhow::ensure!(!prompt_tokens.is_empty(), "expected at least one prompt token");
    let n_prompt = prompt_tokens.len();
    let mut token = prompt_tokens[0] as u32;
    let mut pos = 0u32;
    let mut generated = Vec::new();
    let mut prompt_s = 0.0;
    let mut decode_step_s = Vec::new();
    let mut ttft_s = 0.0;
    let mut first_forward_done = false;
    let mut compat_start = Instant::now();
    let run_start = Instant::now();

    while pos < max_new_tokens.saturating_sub(1) {
        let t0 = Instant::now();
        let logits = model.step(token, pos)?;
        let t1 = Instant::now();
        let next = if (pos as usize) < n_prompt.saturating_sub(1) {
            prompt_s += t1.duration_since(t0).as_secs_f64();
            prompt_tokens[pos as usize + 1]
        } else {
            let nxt = sample_argmax(&logits);
            decode_step_s.push(t1.duration_since(t0).as_secs_f64());
            if decode_step_s.len() == 1 {
                ttft_s = t1.duration_since(run_start).as_secs_f64();
            }
            nxt
        };
        pos += 1;
        if !first_forward_done {
            compat_start = t1;
            first_forward_done = true;
        }
        if stop_on_bos && next == BOS_ID {
            break;
        }
        generated.push(next);
        token = next as u32;
    }

    let compat_elapsed_s = if first_forward_done {
        Instant::now().duration_since(compat_start).as_secs_f64()
    } else {
        0.0
    };
    Ok(LoopOut {
        generated,
        pos,
        prompt_s,
        ttft_s,
        decode_step_s,
        compat_elapsed_s,
    })
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn decode_text(
    tokenizer: &llama3_goldy::tokenizer::Tokenizer,
    prompt_tokens: &[i32],
    generated: &[i32],
) -> String {
    use ammon::Tokenizer as _;
    use llama3_goldy::tokenizer::printable_piece;

    let mut text = String::new();
    let mut prev = prompt_tokens[0];
    for &tok in generated {
        text.push_str(&printable_piece(&tokenizer.decode(prev, tok)));
        prev = tok;
    }
    text
}
