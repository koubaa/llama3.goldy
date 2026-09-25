//! Retained Goldy transformer matching llama3.cuda `forward()`.
//!
//! Ammon records decoder modules into named groups (`embed`, `layerN/attn`,
//! `layerN/ffn`, `tail`) on one retained worker. The `DecodeStep` deposit lives on
//! that submitting root.

use crate::checkpoint::{Checkpoint, Config, ModelShape};
use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::AutoregressiveModel;
use ammon::{CausalAttentionBlock, Embedding, KvCache, Linear, RmsNorm, SwiGluBlock};
use anyhow::{Context, Result};
use goldy::{
    DepositTransaction, FusionReport, HostSink, HostView, MemoryExchange, ReplayStats, Runtime,
    Scheme, Tensor, TensorDType, TensorShape,
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
    attention: CausalAttentionBlock,
    mlp: SwiGluBlock,
    cache: KvCache,
}

impl DecoderModules {
    fn allocate(runtime: &Runtime, shape: &ModelShape) -> Result<Self> {
        Ok(Self {
            attention: CausalAttentionBlock::new(runtime, shape.dim, shape.n_heads, shape.seq_len)?,
            mlp: SwiGluBlock::new(runtime, shape.dim, shape.hidden_dim)?,
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
    _tensors: ModelTensors,
    _modules: DecoderModules,
    worker: Scheme,
    deposit: DepositTransaction,
    logits_sink: HostSink,
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
        // Fused decode is faster once its kernels compile; `GOLDY_FUSION=0` opts out.
        if std::env::var_os("GOLDY_FUSION").is_none() {
            worker.set_automatic_fusion(true);
        }
        let exchange = MemoryExchange::new(&ctx);
        let deposit =
            exchange.bind_deposit(&mut worker, DecodeStep::deposit_target(&tensors.step))?;

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
        let logits_sink = exchange.bind_host_sink(&mut worker, tensors.logits.buffer())?;

        Ok(Self {
            config,
            _tensors: tensors,
            _modules: modules,
            worker,
            deposit,
            logits_sink,
        })
    }

    pub fn step(&mut self, token: u32, pos: u32) -> Result<HostView<f32>> {
        let step = DecodeStep {
            token,
            position: pos,
        };
        (&self.deposit << &step)?;
        let mut submission = self.worker.submit()?;
        // The sink copy is part of the retained scheme and ledger-ordered after
        // the classifier. Claiming only waits and reads its populated staging.
        let claim = &mut submission >> &self.logits_sink;
        Ok(claim.take::<f32>()?)
    }

    pub fn replay_stats(&self) -> ReplayStats {
        self.worker.replay_stats()
    }

    /// Whether the worker's specialization or fusion compiles are outstanding; see
    /// [`Scheme::compiles_pending`].
    pub fn compiles_pending(&self) -> bool {
        self.worker.compiles_pending()
    }

    /// What automatic fusion (on unless `GOLDY_FUSION=0`) runs as one dispatch.
    pub fn fusion_report(&self) -> FusionReport {
        self.worker.fusion_report()
    }

    /// Dispatches and other nodes the worker submits per step, after fusion.
    pub fn executed_node_count(&self) -> usize {
        self.worker.executed_node_count()
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
