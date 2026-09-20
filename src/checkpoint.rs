//! llama3.cuda / llama2.c FP32 checkpoint layout.
//!
//! Binary contract matches `refs/llama3.cuda/llama3.cu` (`Config` + `memory_map_weights`)
//! at commit `424333d1651d2b0fc17d38e9f790e824947e284b`.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Seven-field little-endian header. Negative `vocab_size` means an untied classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Config {
    pub dim: i32,
    pub hidden_dim: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub vocab_size: i32,
    pub max_seq_len: i32,
}

impl Config {
    pub const BYTE_SIZE: usize = std::mem::size_of::<Self>();

    pub fn dim(&self) -> usize {
        usize_field(self.dim, "dim")
    }

    pub fn hidden_dim(&self) -> usize {
        usize_field(self.hidden_dim, "hidden_dim")
    }

    pub fn n_layers(&self) -> usize {
        usize_field(self.n_layers, "n_layers")
    }

    pub fn n_heads(&self) -> usize {
        usize_field(self.n_heads, "n_heads")
    }

    pub fn n_kv_heads(&self) -> usize {
        usize_field(self.n_kv_heads, "n_kv_heads")
    }

    pub fn vocab_size(&self) -> usize {
        usize_field(self.vocab_size, "vocab_size")
    }

    pub fn max_seq_len(&self) -> usize {
        usize_field(self.max_seq_len, "max_seq_len")
    }

    pub fn head_size(&self) -> usize {
        self.dim() / self.n_heads()
    }

    pub fn kv_dim(&self) -> usize {
        self.dim() * self.n_kv_heads() / self.n_heads()
    }

    pub fn kv_mul(&self) -> usize {
        self.n_heads() / self.n_kv_heads()
    }

    pub fn shape(&self) -> ModelShape {
        ModelShape {
            dim: self.dim() as u32,
            hidden_dim: self.hidden_dim() as u32,
            n_layers: self.n_layers() as u32,
            n_heads: self.n_heads() as u32,
            n_kv_heads: self.n_kv_heads() as u32,
            kv_dim: self.kv_dim() as u32,
            kv_mul: self.kv_mul() as u32,
            head_size: self.head_size() as u32,
            seq_len: self.max_seq_len() as u32,
            vocab: self.vocab_size() as u32,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.dim <= 0
            || self.hidden_dim <= 0
            || self.n_layers <= 0
            || self.n_heads <= 0
            || self.n_kv_heads <= 0
            || self.vocab_size <= 0
            || self.max_seq_len <= 0
        {
            bail!("checkpoint config has non-positive fields: {self:?}");
        }
        if self.dim % self.n_heads != 0 {
            bail!(
                "dim {} is not divisible by n_heads {}",
                self.dim,
                self.n_heads
            );
        }
        if self.n_heads % self.n_kv_heads != 0 {
            bail!(
                "n_heads {} is not divisible by n_kv_heads {}",
                self.n_heads,
                self.n_kv_heads
            );
        }
        Ok(())
    }
}

fn usize_field(value: i32, name: &str) -> usize {
    usize::try_from(value).unwrap_or_else(|_| panic!("{name} overflowed usize: {value}"))
}

/// Derived dimensions used by kernels and the retained graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    pub dim: u32,
    pub hidden_dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub kv_dim: u32,
    pub kv_mul: u32,
    pub head_size: u32,
    pub seq_len: u32,
    pub vocab: u32,
}

/// Packed-blob element offsets for one transformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerWeightOffsets {
    pub rms_att: u32,
    pub wq: u32,
    pub wk: u32,
    pub wv: u32,
    pub wo: u32,
    pub rms_ffn: u32,
    pub w1: u32,
    pub w2: u32,
    pub w3: u32,
}

/// Element offsets into the packed FP32 weight blob (not bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightLayout {
    pub token_embedding: u64,
    pub rms_att_weight: u64,
    pub wq: u64,
    pub wk: u64,
    pub wv: u64,
    pub wo: u64,
    pub rms_ffn_weight: u64,
    pub w1: u64,
    pub w2: u64,
    pub w3: u64,
    pub rms_final_weight: u64,
    pub wcls: u64,
    pub n_floats: u64,
    pub shared_classifier: bool,
}

impl WeightLayout {
    pub fn from_config(config: &Config, shared_classifier: bool) -> Result<Self> {
        config.validate()?;
        let dim = config.dim() as u64;
        let hidden_dim = config.hidden_dim() as u64;
        let n_layers = config.n_layers() as u64;
        let vocab = config.vocab_size() as u64;
        let head_size = config.head_size() as u64;
        let max_seq_len = config.max_seq_len() as u64;
        let q_dim = config.n_heads() as u64 * head_size;
        let kv_dim = config.n_kv_heads() as u64 * head_size;

        let mut ptr = 0u64;
        let token_embedding = ptr;
        ptr += vocab * dim;
        let rms_att_weight = ptr;
        ptr += n_layers * dim;
        let wq = ptr;
        ptr += n_layers * dim * q_dim;
        let wk = ptr;
        ptr += n_layers * dim * kv_dim;
        let wv = ptr;
        ptr += n_layers * dim * kv_dim;
        let wo = ptr;
        ptr += n_layers * q_dim * dim;
        let rms_ffn_weight = ptr;
        ptr += n_layers * dim;
        let w1 = ptr;
        ptr += n_layers * dim * hidden_dim;
        let w2 = ptr;
        ptr += n_layers * hidden_dim * dim;
        let w3 = ptr;
        ptr += n_layers * dim * hidden_dim;
        let rms_final_weight = ptr;
        ptr += dim;
        // Skip legacy freq_cis_real / freq_cis_imag slots (RoPE is computed on the fly).
        ptr += max_seq_len * head_size / 2;
        ptr += max_seq_len * head_size / 2;
        let wcls = if shared_classifier {
            token_embedding
        } else {
            let off = ptr;
            ptr += vocab * dim;
            off
        };

        Ok(Self {
            token_embedding,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
            n_floats: ptr,
            shared_classifier,
        })
    }

    pub fn layer(&self, layer: usize, shape: &ModelShape) -> LayerWeightOffsets {
        let l = layer as u64;
        let dim = u64::from(shape.dim);
        let hidden = u64::from(shape.hidden_dim);
        let q = u64::from(shape.n_heads) * u64::from(shape.head_size);
        let kv = u64::from(shape.kv_dim);
        LayerWeightOffsets {
            rms_att: u32_offset(self.rms_att_weight + l * dim),
            wq: u32_offset(self.wq + l * dim * q),
            wk: u32_offset(self.wk + l * dim * kv),
            wv: u32_offset(self.wv + l * dim * kv),
            wo: u32_offset(self.wo + l * q * dim),
            rms_ffn: u32_offset(self.rms_ffn_weight + l * dim),
            w1: u32_offset(self.w1 + l * dim * hidden),
            w2: u32_offset(self.w2 + l * hidden * dim),
            w3: u32_offset(self.w3 + l * dim * hidden),
        }
    }
}

fn u32_offset(off: u64) -> u32 {
    u32::try_from(off).expect("weight element offset exceeds u32")
}

/// Host-side checkpoint: validated header plus the packed FP32 blob.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub config: Config,
    pub layout: WeightLayout,
    pub weights: Vec<f32>,
}

impl Checkpoint {
    pub fn read_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file =
            File::open(path).with_context(|| format!("open checkpoint {}", path.display()))?;
        let mut header = [0u8; Config::BYTE_SIZE];
        file.read_exact(&mut header)
            .with_context(|| format!("read config header from {}", path.display()))?;
        let mut raw: Config = *bytemuck::from_bytes(&header);
        let shared_classifier = raw.vocab_size > 0;
        raw.vocab_size = raw.vocab_size.abs();
        let layout = WeightLayout::from_config(&raw, shared_classifier)?;

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .with_context(|| format!("read weight blob from {}", path.display()))?;
        let expected = layout.n_floats as usize * std::mem::size_of::<f32>();
        if bytes.len() != expected {
            bail!(
                "{}: weight blob is {} bytes, expected {} ({} f32s) for {:?}",
                path.display(),
                bytes.len(),
                expected,
                layout.n_floats,
                raw
            );
        }
        if bytes.len() % 4 != 0 {
            bail!(
                "{}: weight blob is not a multiple of 4 bytes",
                path.display()
            );
        }
        let weights = bytemuck::cast_slice::<u8, f32>(&bytes).to_vec();
        Ok(Self {
            config: raw,
            layout,
            weights,
        })
    }

    pub fn from_parts(
        mut config: Config,
        shared_classifier: bool,
        weights: Vec<f32>,
    ) -> Result<Self> {
        if !shared_classifier {
            config.vocab_size = config.vocab_size.abs();
        }
        let layout = WeightLayout::from_config(&config, shared_classifier)?;
        if weights.len() as u64 != layout.n_floats {
            bail!(
                "weight vec has {} floats, layout expects {}",
                weights.len(),
                layout.n_floats
            );
        }
        Ok(Self {
            config,
            layout,
            weights,
        })
    }
}

/// Stories15M (Karpathy TinyStories) published SHA-256.
pub const STORIES15M_SHA256: &str =
    "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";

/// Pinned llama3.cuda commit this crate replicates.
pub const LLAMA3_CUDA_COMMIT: &str = "424333d1651d2b0fc17d38e9f790e824947e284b";

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tiny_config() -> Config {
        Config {
            dim: 4,
            hidden_dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 4,
            max_seq_len: 4,
        }
    }

    #[test]
    fn tied_layout_matches_pointer_walk() {
        let cfg = tiny_config();
        let layout = WeightLayout::from_config(&cfg, true).unwrap();
        let dim = 4u64;
        let hidden = 8u64;
        let vocab = 4u64;
        let n_layers = 1u64;
        let head_size = 2u64;
        let mut ptr = 0u64;
        assert_eq!(layout.token_embedding, ptr);
        ptr += vocab * dim;
        assert_eq!(layout.rms_att_weight, ptr);
        ptr += n_layers * dim;
        assert_eq!(layout.wq, ptr);
        ptr += n_layers * dim * dim;
        assert_eq!(layout.wk, ptr);
        ptr += n_layers * dim * dim;
        assert_eq!(layout.wv, ptr);
        ptr += n_layers * dim * dim;
        assert_eq!(layout.wo, ptr);
        ptr += n_layers * dim * dim;
        assert_eq!(layout.rms_ffn_weight, ptr);
        ptr += n_layers * dim;
        assert_eq!(layout.w1, ptr);
        ptr += n_layers * dim * hidden;
        assert_eq!(layout.w2, ptr);
        ptr += n_layers * hidden * dim;
        assert_eq!(layout.w3, ptr);
        ptr += n_layers * dim * hidden;
        assert_eq!(layout.rms_final_weight, ptr);
        ptr += dim;
        ptr += 4 * head_size / 2;
        ptr += 4 * head_size / 2;
        assert_eq!(layout.wcls, 0);
        assert_eq!(layout.n_floats, ptr);
        assert!(layout.shared_classifier);

        let shape = cfg.shape();
        let layer0 = layout.layer(0, &shape);
        assert_eq!(layer0.rms_att, layout.rms_att_weight as u32);
        assert_eq!(layer0.wq, layout.wq as u32);
        assert_eq!(layer0.w3, layout.w3 as u32);
    }

    #[test]
    fn layer_offsets_stride_by_tensor_size() {
        let mut cfg = tiny_config();
        cfg.n_layers = 2;
        let layout = WeightLayout::from_config(&cfg, true).unwrap();
        let shape = cfg.shape();
        let l0 = layout.layer(0, &shape);
        let l1 = layout.layer(1, &shape);
        assert_eq!(l1.rms_att, l0.rms_att + shape.dim);
        assert_eq!(l1.wq, l0.wq + shape.dim * shape.dim);
        assert_eq!(l1.wk, l0.wk + shape.dim * shape.kv_dim);
        assert_eq!(l1.w2, l0.w2 + shape.hidden_dim * shape.dim);
        assert_eq!(l1.w3, l0.w3 + shape.dim * shape.hidden_dim);
    }

    #[test]
    fn negative_vocab_size_means_untied_classifier() {
        let dir = std::env::temp_dir();
        let path = dir.join("llama3_goldy_untied.bin");
        let mut cfg = tiny_config();
        let tied = WeightLayout::from_config(&cfg, true).unwrap();
        let untied = WeightLayout::from_config(&cfg, false).unwrap();
        assert_eq!(
            untied.n_floats,
            tied.n_floats + (cfg.vocab_size() * cfg.dim()) as u64
        );
        assert_ne!(untied.wcls, untied.token_embedding);

        cfg.vocab_size = -cfg.vocab_size;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(bytemuck::bytes_of(&cfg));
        bytes.extend(vec![0u8; untied.n_floats as usize * 4]);
        let mut f = File::create(&path).unwrap();
        f.write_all(&bytes).unwrap();
        drop(f);

        let ckpt = Checkpoint::read_path(&path).unwrap();
        assert_eq!(ckpt.config.vocab_size, 4);
        assert!(!ckpt.layout.shared_classifier);
        assert_eq!(ckpt.weights.len() as u64, untied.n_floats);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_truncated_blob() {
        let dir = std::env::temp_dir();
        let path = dir.join("llama3_goldy_trunc.bin");
        let cfg = tiny_config();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(bytemuck::bytes_of(&cfg));
        bytes.extend_from_slice(&[0u8; 16]);
        std::fs::write(&path, bytes).unwrap();
        let err = Checkpoint::read_path(&path).unwrap_err();
        assert!(err.to_string().contains("expected"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_indivisible_heads() {
        let mut cfg = tiny_config();
        cfg.n_heads = 3;
        assert!(WeightLayout::from_config(&cfg, true).is_err());
    }
}
