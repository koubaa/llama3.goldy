//! llama3.cuda host compatibility over Ammon's BPE tokenizer.

pub use ammon::{sample_argmax, BpeTokenizer as Tokenizer};

/// llama3.cuda prompt compatibility patch for `"I have a dream"`.
pub fn apply_dream_prompt_patch(tokens: &mut [i32]) {
    if tokens.len() > 1 && tokens[1] == 306 {
        tokens[1] = 76;
    }
}

/// Match `safe_printf` in llama3.cuda: skip empty / non-printable single bytes,
/// plus the CJK byte-pair rewrite.
pub fn printable_piece(piece: &str) -> String {
    if piece.is_empty() {
        return String::new();
    }
    let bytes = piece.as_bytes();
    if bytes.len() == 1 {
        let b = bytes[0];
        if !(b.is_ascii_graphic() || b.is_ascii_whitespace()) {
            return String::new();
        }
    }
    if bytes.len() >= 2 {
        match bytes[0] {
            0xC3 => return ((bytes[1] | 0x40) as char).to_string(),
            0xC2 => return (bytes[1] as char).to_string(),
            _ => {}
        }
    }
    piece.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dream_patch_rewrites_token_one() {
        let mut ids = vec![1, 306, 40];
        apply_dream_prompt_patch(&mut ids);
        assert_eq!(ids[1], 76);
        let mut other = vec![1, 77];
        apply_dream_prompt_patch(&mut other);
        assert_eq!(other[1], 77);
    }
}
