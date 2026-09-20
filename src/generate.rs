//! Prompt processing, greedy decode, and llama3.cuda host compatibility.

use crate::model::Model;
use crate::tokenizer::{apply_dream_prompt_patch, printable_piece, Tokenizer};
use ammon::Tokenizer as _;
use anyhow::Result;

/// llama3.cuda README output for `"I have a dream"` at 50 tokens on stories15M.
pub const DREAM_STORY: &str = "\
I have a dream. He dreams of a big, beautiful garden full of flowers and trees. He dreams of playing with his friends and eating yummy snacks.\n\
One day, he was walking in the garden when he saw";

#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub text: String,
    pub tokens: Vec<i32>,
    pub prompt_tokens: usize,
    pub worker_records: u64,
    pub tokens_per_second: f64,
}

pub fn generate(
    model: &mut Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: u32,
) -> Result<GenerateOutput> {
    let mut prompt_tokens = tokenizer.encode(prompt, true, false);
    apply_dream_prompt_patch(&mut prompt_tokens);
    let out = ammon::generate_tokens(
        model,
        &prompt_tokens,
        max_new_tokens,
        Some(Tokenizer::bos_id()),
    )?;

    let mut text = String::new();
    let mut prev = prompt_tokens[0];
    for &tok in &out.tokens {
        let piece = tokenizer.decode(prev, tok);
        text.push_str(&printable_piece(&piece));
        prev = tok;
    }

    Ok(GenerateOutput {
        text,
        tokens: out.tokens,
        prompt_tokens: out.prompt_tokens,
        worker_records: model.replay_stats().records,
        tokens_per_second: out.tokens_per_second,
    })
}
