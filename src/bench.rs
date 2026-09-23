//! Shared TinyStories benchmark protocol (no GPU required).

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// llama3.cuda README output for `"I have a dream"` at 50 tokens on stories15M.
pub const DREAM_STORY: &str = "\
I have a dream. He dreams of a big, beautiful garden full of flowers and trees. He dreams of playing with his friends and eating yummy snacks.\n\
One day, he was walking in the garden when he saw";

pub const DREAM_PROMPT: &str = "I have a dream";
pub const COMPAT_TOTAL_POSITIONS: u32 = 50;
pub const SCALING_CONTEXT_LENGTHS: [u32; 4] = [8, 32, 128, 224];
pub const SCALING_DECODE_STEPS: u32 = 16;
pub const BOS_ID: i32 = 1;
pub const SCHEMA_VERSION: u32 = 1;

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn expected_checkpoint_sha256(name: &str) -> Option<&'static str> {
    match name {
        "stories15M.bin" => Some(crate::STORIES15M_SHA256),
        "stories42M.bin" => Some(crate::STORIES42M_SHA256),
        "stories110M.bin" => Some(crate::STORIES110M_SHA256),
        "tokenizer.bin" => Some(crate::TOKENIZER_SHA256),
        _ => None,
    }
}

/// Truncate or pad prompt tokens to `length`. Padding never uses BOS.
pub fn pad_or_trim(tokens: &[i32], length: usize) -> Result<Vec<i32>> {
    anyhow::ensure!(length > 0, "context length must be positive");
    if tokens.len() >= length {
        return Ok(tokens[..length].to_vec());
    }
    let filler = tokens
        .iter()
        .rev()
        .copied()
        .find(|&t| t != BOS_ID)
        .unwrap_or(13);
    let mut out = tokens.to_vec();
    out.resize(length, filler);
    Ok(out)
}

pub fn rates(
    prompt_tokens: usize,
    prompt_s: f64,
    decode_n: usize,
    decode_s: f64,
    pos: u32,
    elapsed: f64,
) -> (f64, f64, f64) {
    let prompt_tok_s = if prompt_s > 0.0 {
        (prompt_tokens.saturating_sub(1) as f64) / prompt_s
    } else {
        0.0
    };
    let decode_tok_s = if decode_s > 0.0 {
        decode_n as f64 / decode_s
    } else {
        0.0
    };
    let legacy = if elapsed > 0.0 {
        (pos.saturating_sub(1) as f64) / elapsed
    } else {
        0.0
    };
    (prompt_tok_s, decode_tok_s, legacy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_trims_and_avoids_bos_filler() {
        assert_eq!(pad_or_trim(&[1, 76, 40, 13], 2).unwrap(), vec![1, 76]);
        assert_eq!(
            pad_or_trim(&[1, 76], 4).unwrap(),
            vec![1, 76, 76, 76]
        );
        assert_eq!(pad_or_trim(&[1], 3).unwrap(), vec![1, 13, 13]);
    }

    #[test]
    fn sha256_matches_known_abc() {
        let dir = std::env::temp_dir();
        let path = dir.join("llama3_goldy_sha_abc.bin");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pinned_names_have_sha() {
        assert!(expected_checkpoint_sha256("stories15M.bin").is_some());
        assert!(expected_checkpoint_sha256("stories110M.bin").is_some());
        assert!(expected_checkpoint_sha256("mystery.bin").is_none());
    }
}
