//! Retained Goldy transformer matching llama3.cuda `forward()`.
//!
//! Ammon records decoder modules into named groups (`embed`, `layerN/attn`,
//! `layerN/ffn`, `tail`) on one retained worker. The `DecodeStep` deposit lives on
//! that submitting root.

use crate::checkpoint::{Checkpoint, Config, ModelShape};
use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::AutoregressiveModel;
use ammon::{CausalSelfAttention, Embedding, KvCache, Linear, RmsNorm, SwiGluMlp};
use anyhow::{Context, Result};
use goldy::{
    DepositTransaction, HostView, MemoryExchange, ReplayStats, Runtime, Scheme, Tensor,
    TensorDType, TensorShape,
};

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
            step: DecodeStep::parcel(runtime)?,
            logits: Tensor::zeros(runtime, TensorShape::vector(shape.vocab), TensorDType::F32)?,
        })
    }
}

/// Scratch the worker retains. Kept alive for the scheme lifetime.
struct DecoderModules {
    attention: CausalSelfAttention,
    mlp: SwiGluMlp,
    cache: KvCache,
}

impl DecoderModules {
    fn allocate(runtime: &Runtime, shape: &ModelShape) -> Result<Self> {
        Ok(Self {
            attention: CausalSelfAttention::new(runtime, shape.dim, shape.n_heads, shape.seq_len)?,
            mlp: SwiGluMlp::new(runtime, shape.dim, shape.hidden_dim)?,
            cache: KvCache::new(
                runtime,
                shape.n_layers,
                shape.seq_len,
                shape.n_kv_heads,
                shape.head_size,
            )?,
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
        Self::load_on(&runtime, checkpoint)
    }

    pub fn load_on(runtime: &Runtime, checkpoint: &Checkpoint) -> Result<Self> {
        let ctx = runtime.create_context().context("create GPU context")?;
        let config = checkpoint.config;
        let shape = config.shape();
        let layout = checkpoint.layout;
        let modules = DecoderModules::allocate(runtime, &shape)?;
        let tensors = ModelTensors::allocate(runtime, checkpoint, &shape)?;

        let mut worker = Scheme::new(&ctx);
        let deposit = MemoryExchange::new(&ctx)
            .bind_deposit(&mut worker, DecodeStep::deposit_target(&tensors.step))?;

        Embedding::new(runtime)?.record_group(
            &mut worker,
            "embed",
            layout.embedding(&tensors.weights, &shape)?,
            &tensors.step,
            tensors.x.view(),
        )?;

        for layer in 0..shape.n_layers {
            let weights = layout.layer(&tensors.weights, layer as usize, &shape)?;
            modules.attention.record_group(
                &mut worker,
                format!("layer{layer}/attn"),
                tensors.x.view(),
                weights.attention,
                modules.cache.layer(layer)?,
                &tensors.step,
            )?;
            modules.mlp.record_group(
                &mut worker,
                format!("layer{layer}/ffn"),
                tensors.x.view(),
                weights.mlp,
            )?;
        }

        let rms_final = layout.rms_final(&tensors.weights, &shape)?;
        let classifier = layout.classifier(&tensors.weights, &shape)?;
        let norm = RmsNorm::new(runtime)?;
        let lm_head = Linear::new(runtime)?;
        worker.group("tail", |scheme| {
            norm.record_inplace(scheme, tensors.x.view(), rms_final)?;
            lm_head.record(scheme, classifier, tensors.x.view(), tensors.logits.view())
        })?;

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
        // `>>` claims mid-flight; `take` waits for the worker then, on CUDA, submits a
        // second copy into staging and waits again. That extra round trip is not interned
        // on the scheme the way the deposit is.
        let claim = &mut submission >> self.tensors.logits.buffer();
        Ok(claim.take::<f32>()?)
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
