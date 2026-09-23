//! Retained Goldy transformer matching llama3.cuda `forward()`.
//!
//! Ammon records decoder modules into named groups (`embed`, `layerN/attn`,
//! `layerN/ffn`, `tail`) on one retained worker. The `DecodeStep` deposit lives on
//! that submitting root.

use crate::checkpoint::{Checkpoint, Config, LayerWeightViews, ModelShape};
use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::AutoregressiveModel;
use ammon::{AttentionWeights, Blocks, CausalSelfAttention, KvCache, SwiGluMlp, SwiGluWeights};
use anyhow::{Context, Result};
use goldy::{
    BufferKind, DepositTarget, DepositTransaction, HostView, MemoryExchange, ReplayStats, Runtime,
    Scheme, Tensor, TensorDType, TensorShape,
};
use std::sync::Arc;

struct ModelTensors {
    weights: Tensor,
    x: Tensor,
    step: goldy::Buffer,
    logits: Tensor,
}

impl ModelTensors {
    fn allocate(runtime: &Runtime, checkpoint: &Checkpoint, shape: &ModelShape) -> Result<Self> {
        let n_weights =
            u32::try_from(checkpoint.weights.len()).context("weight count exceeds u32")?;
        Ok(Self {
            weights: Tensor::from_f32(runtime, TensorShape::vector(n_weights), &checkpoint.weights)
                .context("upload weights")?,
            x: Tensor::zeros(runtime, TensorShape::vector(shape.dim), TensorDType::F32)?,
            step: runtime.acquire_buffer_with_data(
                &[DecodeStep {
                    token: 0,
                    position: 0,
                }],
                BufferKind::Scattered,
            )?,
            logits: Tensor::zeros(runtime, TensorShape::vector(shape.vocab), TensorDType::F32)?,
        })
    }
}

/// Scratch and kernels the worker retains. Kept alive for the scheme lifetime.
struct DecoderModules {
    _blocks: Arc<Blocks>,
    attention: CausalSelfAttention,
    mlp: SwiGluMlp,
    cache: KvCache,
}

impl DecoderModules {
    fn allocate(runtime: &Runtime, shape: &ModelShape) -> Result<Self> {
        let blocks = Arc::new(Blocks::prepare(runtime)?);
        Ok(Self {
            attention: CausalSelfAttention::with_blocks(
                runtime,
                Arc::clone(&blocks),
                shape.dim,
                shape.n_heads,
                shape.seq_len,
            )?,
            mlp: SwiGluMlp::with_blocks(runtime, Arc::clone(&blocks), shape.dim, shape.hidden_dim)?,
            cache: KvCache::new(
                runtime,
                shape.n_layers,
                shape.seq_len,
                shape.n_kv_heads,
                shape.head_size,
            )?,
            _blocks: blocks,
        })
    }
}

pub struct Model {
    pub config: Config,
    tensors: ModelTensors,
    _modules: DecoderModules,
    worker: Scheme,
    deposit: DepositTransaction,
}

impl Model {
    pub fn load(checkpoint: &Checkpoint) -> Result<Self> {
        let runtime = create_runtime()?;
        Self::load_on(runtime, checkpoint)
    }

    pub fn load_on(runtime: Runtime, checkpoint: &Checkpoint) -> Result<Self> {
        let ctx = runtime.create_context().context("create GPU context")?;
        Self::build(runtime, ctx, checkpoint)
    }

    fn build(runtime: Runtime, ctx: goldy::Context, checkpoint: &Checkpoint) -> Result<Self> {
        let config = checkpoint.config;
        let shape = config.shape();
        let layout = checkpoint.layout;
        let modules = DecoderModules::allocate(&runtime, &shape)?;
        let tensors = ModelTensors::allocate(&runtime, checkpoint, &shape)?;

        let mut worker = Scheme::new(&ctx);
        let deposit = MemoryExchange::new(&ctx).bind_deposit(
            &mut worker,
            DepositTarget::buffer_elements::<DecodeStep>(&tensors.step, 1),
        )?;

        let embedding = layout.embedding(&tensors.weights, &shape)?;
        modules._blocks.record_embed_group(
            &mut worker,
            "embed",
            embedding,
            &tensors.step,
            tensors.x.view(),
        )?;

        for layer in 0..shape.n_layers as usize {
            let views = layout.layer_views(&tensors.weights, layer, &shape)?;
            modules.attention.record_group(
                &mut worker,
                format!("layer{layer}/attn"),
                tensors.x.view(),
                attention_weights(&views),
                modules.cache.layer(layer)?,
                &tensors.step,
            )?;
            modules.mlp.record_group(
                &mut worker,
                format!("layer{layer}/ffn"),
                tensors.x.view(),
                swiglu_weights(&views),
            )?;
        }

        let rms_final = layout.rms_final(&tensors.weights, &shape)?;
        let classifier = layout.classifier(&tensors.weights, &shape)?;
        modules._blocks.record_logits_group(
            &mut worker,
            "tail",
            tensors.x.view(),
            rms_final,
            classifier,
            tensors.logits.view(),
        )?;

        Ok(Self {
            config,
            tensors,
            _modules: modules,
            worker,
            deposit,
        })
    }

    pub fn step(&mut self, token: u32, pos: u32) -> Result<HostView<f32>> {
        let step = DecodeStep {
            token,
            position: pos,
        };
        (&self.deposit << &step)?;
        let mut submission = self.worker.submit()?;
        let logits = (&mut submission >> self.tensors.logits.buffer()).take::<f32>()?;
        anyhow::ensure!(
            logits.len() == self.config.vocab_size(),
            "logit withdraw size {} != vocab {}",
            logits.len(),
            self.config.vocab_size()
        );
        Ok(logits)
    }

    pub fn replay_stats(&self) -> ReplayStats {
        self.worker.replay_stats()
    }
}

impl AutoregressiveModel for Model {
    fn vocab_size(&self) -> usize {
        self.config.vocab_size()
    }

    fn max_seq_len(&self) -> u32 {
        self.config.max_seq_len() as u32
    }

    fn step(&mut self, token: u32, pos: u32) -> Result<HostView<f32>> {
        Model::step(self, token, pos)
    }
}

fn attention_weights<'a>(weights: &LayerWeightViews<'a>) -> AttentionWeights<'a> {
    AttentionWeights {
        norm: weights.rms_att,
        query: weights.wq,
        key: weights.wk,
        value: weights.wv,
        output: weights.wo,
    }
}

fn swiglu_weights<'a>(weights: &LayerWeightViews<'a>) -> SwiGluWeights<'a> {
    SwiGluWeights {
        norm: weights.rms_ffn,
        gate: weights.w1,
        up: weights.w3,
        down: weights.w2,
    }
}
