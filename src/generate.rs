//! Prompt processing, greedy decode, and host-side control loop.

use crate::model::Model;
use crate::tokenizer::{sample_argmax, Tokenizer};
use anyhow::Result;
use std::time::Instant;

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
    Tokenizer::apply_dream_prompt_patch(&mut prompt_tokens);
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "expected at least one prompt token"
    );

    let max_new_tokens = max_new_tokens.min(model.config.max_seq_len() as u32);
    let mut text = String::new();
    let mut tokens = Vec::new();
    let mut token = prompt_tokens[0] as u32;
    let mut pos = 0u32;
    let mut start = None;

    while pos < max_new_tokens.saturating_sub(1) {
        let logits = model.step(token, pos)?;
        let next = if (pos as usize) < prompt_tokens.len() - 1 {
            prompt_tokens[pos as usize + 1]
        } else {
            sample_argmax(&logits)
        };
        pos += 1;
        if next == 1 {
            break;
        }
        let piece = tokenizer.decode(token as i32, next);
        text.push_str(&Tokenizer::printable_piece(&piece));
        tokens.push(next);
        token = next as u32;
        if start.is_none() {
            start = Some(Instant::now());
        }
    }

    let elapsed = start.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
    let gen_tokens = pos.saturating_sub(1) as f64;
    let tokens_per_second = if elapsed > 0.0 {
        gen_tokens / elapsed
    } else {
        0.0
    };

    Ok(GenerateOutput {
        text,
        tokens,
        prompt_tokens: prompt_tokens.len(),
        worker_records: model.replay_stats().records,
        tokens_per_second,
    })
}
