//! llama3.cuda TinyStories generator on Goldy (CUDA / Metal).

pub mod checkpoint;
pub mod tokenizer;

#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod generate;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod gpu;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod kernels;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod model;

pub use checkpoint::{Checkpoint, Config, WeightLayout, LLAMA3_CUDA_COMMIT, STORIES15M_SHA256};
pub use tokenizer::{sample_argmax, Tokenizer};
