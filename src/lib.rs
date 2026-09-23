//! llama3.cuda TinyStories generator on Ammon / Goldy (CUDA / Metal).

pub mod checkpoint;
pub mod tokenizer;

#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod generate;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod model;

#[cfg(any(feature = "cuda", feature = "metal"))]
pub use checkpoint::LayerWeights;
pub use checkpoint::{
    Checkpoint, Config, LayerWeightOffsets, ModelShape, WeightLayout, LLAMA3_CUDA_COMMIT,
    STORIES15M_SHA256,
};
pub use tokenizer::{apply_dream_prompt_patch, printable_piece, Tokenizer};
