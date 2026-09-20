//! Retained Goldy transformer matching llama3.cuda `forward()`.

use crate::checkpoint::{Checkpoint, Config, LayerWeightOffsets, ModelShape, WeightLayout};
use crate::gpu::create_runtime;
use crate::kernels::{
    AttentionKernel, DecodeStep, EmbedKernel, GemvKernel, ResidualAddKernel, RmsnormInplaceKernel,
    RmsnormKernel, RopeKernel, SwigluKernel,
};
use anyhow::{Context, Result};
use goldy::{
    Buffer, BufferKind, Context as GpuContext, DepositTarget, DepositTransaction, MemoryExchange,
    ReplayStats, Runtime, Scheme, WithdrawTransaction,
};

struct PreparedKernels {
    embed: EmbedKernel,
    rmsnorm: RmsnormKernel,
    rmsnorm_inplace: RmsnormInplaceKernel,
    gemv: GemvKernel,
    rope: RopeKernel,
    attention: AttentionKernel,
    swiglu: SwigluKernel,
    residual_add: ResidualAddKernel,
}

impl PreparedKernels {
    fn prepare(runtime: &Runtime) -> Result<Self> {
        Ok(Self {
            embed: EmbedKernel::prepare(runtime).context("prepare embed kernel")?,
            rmsnorm: RmsnormKernel::prepare(runtime).context("prepare rmsnorm kernel")?,
            rmsnorm_inplace: RmsnormInplaceKernel::prepare(runtime)
                .context("prepare rmsnorm_inplace kernel")?,
            gemv: GemvKernel::prepare(runtime).context("prepare gemv kernel")?,
            rope: RopeKernel::prepare(runtime).context("prepare rope kernel")?,
            attention: AttentionKernel::prepare(runtime).context("prepare attention kernel")?,
            swiglu: SwigluKernel::prepare(runtime).context("prepare swiglu kernel")?,
            residual_add: ResidualAddKernel::prepare(runtime)
                .context("prepare residual_add kernel")?,
        })
    }
}

struct ModelBuffers {
    weights: Buffer,
    x: Buffer,
    xb: Buffer,
    xb2: Buffer,
    hb: Buffer,
    hb2: Buffer,
    q: Buffer,
    att: Buffer,
    key_cache: Buffer,
    value_cache: Buffer,
    step: Buffer,
    logits: Buffer,
    #[allow(dead_code)]
    runtime: Runtime,
}

impl ModelBuffers {
    fn allocate(runtime: Runtime, checkpoint: &Checkpoint, shape: &ModelShape) -> Result<Self> {
        let zeros = |n: usize| vec![0f32; n];
        let dim = shape.dim as usize;
        let hidden = shape.hidden_dim as usize;
        let kv_elems = (shape.n_layers * shape.seq_len * shape.kv_dim) as usize;
        let att_elems = (shape.n_heads * shape.seq_len) as usize;
        Ok(Self {
            weights: runtime
                .acquire_buffer_with_data(&checkpoint.weights, BufferKind::Scattered)
                .context("upload weights")?,
            x: runtime.acquire_buffer_with_data(&zeros(dim), BufferKind::Scattered)?,
            xb: runtime.acquire_buffer_with_data(&zeros(dim), BufferKind::Scattered)?,
            xb2: runtime.acquire_buffer_with_data(&zeros(dim), BufferKind::Scattered)?,
            hb: runtime.acquire_buffer_with_data(&zeros(hidden), BufferKind::Scattered)?,
            hb2: runtime.acquire_buffer_with_data(&zeros(hidden), BufferKind::Scattered)?,
            q: runtime.acquire_buffer_with_data(&zeros(dim), BufferKind::Scattered)?,
            att: runtime.acquire_buffer_with_data(&zeros(att_elems), BufferKind::Scattered)?,
            key_cache: runtime.acquire_buffer_with_data(&zeros(kv_elems), BufferKind::Scattered)?,
            value_cache: runtime
                .acquire_buffer_with_data(&zeros(kv_elems), BufferKind::Scattered)?,
            step: runtime.acquire_buffer_with_data(
                &[DecodeStep {
                    token: 0,
                    position: 0,
                }],
                BufferKind::Scattered,
            )?,
            logits: runtime
                .acquire_buffer_with_data(&zeros(shape.vocab as usize), BufferKind::Scattered)?,
            runtime,
        })
    }
}

pub struct Model {
    pub config: Config,
    _buffers: ModelBuffers,
    _kernels: PreparedKernels,
    worker: Scheme,
    upload: Scheme,
    deposit: DepositTransaction,
    withdraw: WithdrawTransaction,
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

    fn build(runtime: Runtime, ctx: GpuContext, checkpoint: &Checkpoint) -> Result<Self> {
        let config = checkpoint.config;
        let shape = config.shape();
        let layout = checkpoint.layout;
        let kernels = PreparedKernels::prepare(&runtime)?;
        let buffers = ModelBuffers::allocate(runtime, checkpoint, &shape)?;

        let mut worker = Scheme::new(&ctx);
        kernels
            .embed
            .record(
                &mut worker,
                "embed",
                &buffers.weights,
                &buffers.step,
                &buffers.x,
                shape.dim,
            )
            .over_1d(shape.dim);

        for layer in 0..shape.n_layers as usize {
            record_layer(&mut worker, &kernels, &buffers, &shape, &layout, layer);
        }

        kernels
            .rmsnorm_inplace
            .record(
                &mut worker,
                "rmsnorm_final",
                &buffers.x,
                &buffers.weights,
                shape.dim,
                u32::try_from(layout.rms_final_weight).unwrap(),
            )
            .groups([1, 1, 1]);

        kernels
            .gemv
            .record(
                &mut worker,
                "classifier",
                &buffers.x,
                &buffers.weights,
                &buffers.logits,
                &buffers.step,
                shape.dim,
                shape.vocab,
                u32::try_from(layout.wcls).unwrap(),
                0,
                0,
            )
            .over_1d(shape.vocab);

        let memory = MemoryExchange::new(&ctx);
        let withdraw = memory.bind_withdraw(&mut worker, &buffers.logits)?;

        let mut upload = Scheme::new(&ctx);
        let deposit = memory.bind_deposit(
            &mut upload,
            DepositTarget::buffer_elements::<DecodeStep>(&buffers.step, 1),
        )?;

        Ok(Self {
            config,
            _buffers: buffers,
            _kernels: kernels,
            worker,
            upload,
            deposit,
            withdraw,
        })
    }

    pub fn step(&mut self, token: u32, pos: u32) -> Result<Vec<f32>> {
        self.deposit.write_data(
            0,
            &[DecodeStep {
                token,
                position: pos,
            }],
        )?;
        let _ = self.upload.submit()?;
        let mut submission = self.worker.submit()?;
        let bytes = self.withdraw.claim(&mut submission)?.consume()?;
        let logits: &[f32] = bytemuck::cast_slice(&bytes);
        anyhow::ensure!(
            logits.len() == self.config.vocab_size(),
            "logit withdraw size {} != vocab {}",
            logits.len(),
            self.config.vocab_size()
        );
        Ok(logits.to_vec())
    }

    pub fn replay_stats(&self) -> ReplayStats {
        self.worker.replay_stats()
    }
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn record_layer(
    worker: &mut Scheme,
    kernels: &PreparedKernels,
    buffers: &ModelBuffers,
    shape: &ModelShape,
    layout: &WeightLayout,
    layer: usize,
) {
    let loff = layer as u32 * shape.seq_len * shape.kv_dim;
    let weights = layout.layer(layer, shape);
    record_attention_block(worker, kernels, buffers, shape, &weights, layer, loff);
    record_ffn_block(worker, kernels, buffers, shape, &weights, layer);
}

fn record_attention_block(
    worker: &mut Scheme,
    kernels: &PreparedKernels,
    buffers: &ModelBuffers,
    shape: &ModelShape,
    weights: &LayerWeightOffsets,
    layer: usize,
    loff: u32,
) {
    kernels
        .rmsnorm
        .record(
            worker,
            leak(format!("rmsnorm_att_{layer}")),
            &buffers.x,
            &buffers.weights,
            &buffers.xb,
            shape.dim,
            weights.rms_att,
        )
        .groups([1, 1, 1]);

    kernels
        .gemv
        .record(
            worker,
            leak(format!("wq_{layer}")),
            &buffers.xb,
            &buffers.weights,
            &buffers.q,
            &buffers.step,
            shape.dim,
            shape.dim,
            weights.wq,
            0,
            0,
        )
        .over_1d(shape.dim);

    kernels
        .gemv
        .record(
            worker,
            leak(format!("wk_{layer}")),
            &buffers.xb,
            &buffers.weights,
            &buffers.key_cache,
            &buffers.step,
            shape.dim,
            shape.kv_dim,
            weights.wk,
            loff,
            shape.kv_dim,
        )
        .over_1d(shape.kv_dim);

    kernels
        .gemv
        .record(
            worker,
            leak(format!("wv_{layer}")),
            &buffers.xb,
            &buffers.weights,
            &buffers.value_cache,
            &buffers.step,
            shape.dim,
            shape.kv_dim,
            weights.wv,
            loff,
            shape.kv_dim,
        )
        .over_1d(shape.kv_dim);

    kernels
        .rope
        .record(
            worker,
            leak(format!("rope_{layer}")),
            &buffers.q,
            &buffers.key_cache,
            &buffers.step,
            shape.kv_dim,
            shape.head_size,
            shape.dim,
            loff,
        )
        .over_1d((shape.dim / 2).max(1));

    kernels
        .attention
        .record(
            worker,
            leak(format!("attn_{layer}")),
            &buffers.q,
            &buffers.att,
            &buffers.xb,
            &buffers.key_cache,
            &buffers.value_cache,
            &buffers.step,
            shape.kv_dim,
            shape.kv_mul,
            shape.head_size,
            shape.seq_len,
            loff,
        )
        .groups([shape.n_heads, 1, 1]);

    kernels
        .gemv
        .record(
            worker,
            leak(format!("wo_{layer}")),
            &buffers.xb,
            &buffers.weights,
            &buffers.xb2,
            &buffers.step,
            shape.dim,
            shape.dim,
            weights.wo,
            0,
            0,
        )
        .over_1d(shape.dim);

    kernels
        .residual_add
        .record(
            worker,
            leak(format!("residual_att_{layer}")),
            &buffers.x,
            &buffers.xb2,
            shape.dim,
        )
        .over_1d(shape.dim);
}

fn record_ffn_block(
    worker: &mut Scheme,
    kernels: &PreparedKernels,
    buffers: &ModelBuffers,
    shape: &ModelShape,
    weights: &LayerWeightOffsets,
    layer: usize,
) {
    kernels
        .rmsnorm
        .record(
            worker,
            leak(format!("rmsnorm_ffn_{layer}")),
            &buffers.x,
            &buffers.weights,
            &buffers.xb,
            shape.dim,
            weights.rms_ffn,
        )
        .groups([1, 1, 1]);

    kernels
        .gemv
        .record(
            worker,
            leak(format!("w1_{layer}")),
            &buffers.xb,
            &buffers.weights,
            &buffers.hb,
            &buffers.step,
            shape.dim,
            shape.hidden_dim,
            weights.w1,
            0,
            0,
        )
        .over_1d(shape.hidden_dim);

    kernels
        .gemv
        .record(
            worker,
            leak(format!("w3_{layer}")),
            &buffers.xb,
            &buffers.weights,
            &buffers.hb2,
            &buffers.step,
            shape.dim,
            shape.hidden_dim,
            weights.w3,
            0,
            0,
        )
        .over_1d(shape.hidden_dim);

    kernels
        .swiglu
        .record(
            worker,
            leak(format!("swiglu_{layer}")),
            &buffers.hb,
            &buffers.hb2,
            shape.hidden_dim,
        )
        .over_1d(shape.hidden_dim);

    kernels
        .gemv
        .record(
            worker,
            leak(format!("w2_{layer}")),
            &buffers.hb,
            &buffers.weights,
            &buffers.xb,
            &buffers.step,
            shape.hidden_dim,
            shape.dim,
            weights.w2,
            0,
            0,
        )
        .over_1d(shape.dim);

    kernels
        .residual_add
        .record(
            worker,
            leak(format!("residual_ffn_{layer}")),
            &buffers.x,
            &buffers.xb,
            shape.dim,
        )
        .over_1d(shape.dim);
}
