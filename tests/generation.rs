//! End-to-end greedy generation vs the published llama3.cuda TinyStories sample.

#![cfg(any(feature = "cuda", feature = "metal"))]

use llama3_goldy::checkpoint::Checkpoint;
use llama3_goldy::generate::{generate, DREAM_STORY};
use llama3_goldy::model::Model;
use llama3_goldy::tokenizer::Tokenizer;
use std::path::Path;

fn models_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models")
}

#[test]
fn stories15m_i_have_a_dream() {
    let ckpt_path = models_dir().join("stories15M.bin");
    let tok_path = models_dir().join("tokenizer.bin");
    if !ckpt_path.exists() || !tok_path.exists() {
        eprintln!(
            "skip: fetch assets with `python tools/fetch_assets.py` (missing {} or {})",
            ckpt_path.display(),
            tok_path.display()
        );
        return;
    }

    let ckpt = Checkpoint::read_path(&ckpt_path).expect("checkpoint");
    assert_eq!(ckpt.config.dim, 288);
    assert_eq!(ckpt.config.n_layers, 6);
    assert_eq!(
        ckpt.config.n_kv_heads, ckpt.config.n_heads,
        "stories15M is MHA"
    );
    let tokenizer = Tokenizer::from_path(&tok_path, ckpt.config.vocab_size()).expect("tokenizer");
    let mut model = Model::load(&ckpt).expect("load model on GPU");
    let out = generate(&mut model, &tokenizer, "I have a dream", 50).expect("generate");
    let stats = model.replay_stats();
    eprintln!("generated:\n{}", out.text);
    eprintln!("replay stats: {stats:?}");

    assert_eq!(
        out.text, DREAM_STORY,
        "generated text must match llama3.cuda README sample\n--- got ---\n{}\n--- expected ---\n{}",
        out.text, DREAM_STORY
    );
    assert_eq!(
        stats.topology_records, 0,
        "DecodeStep deposit must not dirty worker topology"
    );
    assert!(
        stats.records <= 2,
        "worker records once, plus at most one specialization re-record; got {stats:?}"
    );
    #[cfg(not(feature = "metal"))]
    assert!(
        stats.resubmit_hits >= 40,
        "steady-state submits should hit retained replay; got {stats:?}"
    );
}
