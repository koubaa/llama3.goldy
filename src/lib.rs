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
    checkpoint_file_bytes, Checkpoint, Config, LayerWeightOffsets, ModelShape, WeightLayout,
    LLAMA3_CUDA_COMMIT, LLAMA_CPP_COMMIT, STORIES110M_SHA256, STORIES15M_SHA256, STORIES42M_SHA256,
    TOKENIZER_SHA256,
};
pub use tokenizer::{apply_dream_prompt_patch, printable_piece, Tokenizer};
