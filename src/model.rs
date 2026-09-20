//! Retained Goldy transformer matching llama3.cuda `forward()`.

use crate::checkpoint::{Checkpoint, Config, LayerWeightViews, ModelShape};
use ammon::gpu::create_runtime;
use ammon::kernels::{
    AttentionKernel, DecodeStep, EmbedKernel, GemvKernel, RmsnormInplaceKernel, RmsnormKernel, RopeKernel,
    SwigluKernel, TensorKernels, DEFAULT_ROPE_THETA,
};
use ammon::AutoregressiveModel;
use anyhow::{Context, Result};
use goldy::{
    BufferKind, Context as GpuContext, DepositTarget, DepositTransaction, MemoryExchange, ReplayStats, Runtime, Scheme,
    Tensor, TensorDType, TensorShape, TensorView, WithdrawTransaction,
};

struct PreparedKernels {
    embed: EmbedKernel,
    rmsnorm: RmsnormKernel,
    rmsnorm_inplace: RmsnormInplaceKernel,
    gemv: GemvKernel,
    rope: RopeKernel,
    attention: AttentionKernel,
    swiglu: SwigluKernel,
    tensors: TensorKernels,
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
            tensors: TensorKernels::prepare(runtime).context("prepare tensor kernels")?,
        })
    }
}

struct ModelTensors {
    weights: Tensor,
    x: Tensor,
    xb: Tensor,
    xb2: Tensor,
    hb: Tensor,
    hb2: Tensor,
    q: Tensor,
    att: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    step: goldy::Buffer,
    logits: Tensor,
    #[allow(dead_code)]
    runtime: Runtime,
}

impl ModelTensors {
    fn allocate(runtime: Runtime, checkpoint: &Checkpoint, shape: &ModelShape) -> Result<Self> {
        let n_weights = u32::try_from(checkpoint.weights.len()).context("weight count exceeds u32")?;
        let kv_elems = shape.n_layers * shape.seq_len * shape.kv_dim;
        let att_elems = shape.n_heads * shape.seq_len;
        Ok(Self {
            weights: Tensor::from_f32(&runtime, TensorShape::vector(n_weights), &checkpoint.weights)
                .context("upload weights")?,
            x: Tensor::zeros(&runtime, TensorShape::vector(shape.dim), TensorDType::F32)?,
            xb: Tensor::zeros(&runtime, TensorShape::vector(shape.dim), TensorDType::F32)?,
            xb2: Tensor::zeros(&runtime, TensorShape::vector(shape.dim), TensorDType::F32)?,
            hb: Tensor::zeros(&runtime, TensorShape::vector(shape.hidden_dim), TensorDType::F32)?,
            hb2: Tensor::zeros(&runtime, TensorShape::vector(shape.hidden_dim), TensorDType::F32)?,
            q: Tensor::zeros(&runtime, TensorShape::vector(shape.dim), TensorDType::F32)?,
            att: Tensor::zeros(&runtime, TensorShape::vector(att_elems), TensorDType::F32)?,
            key_cache: Tensor::zeros(&runtime, TensorShape::vector(kv_elems), TensorDType::F32)?,
            value_cache: Tensor::zeros(&runtime, TensorShape::vector(kv_elems), TensorDType::F32)?,
            step: runtime.acquire_buffer_with_data(
                &[DecodeStep {
                    token: 0,
                    position: 0,
                }],
                BufferKind::Scattered,
            )?,
            logits: Tensor::zeros(&runtime, TensorShape::vector(shape.vocab), TensorDType::F32)?,
            runtime,
        })
    }
}

pub struct Model {
    pub config: Config,
    _tensors: ModelTensors,
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
        let mut kernels = PreparedKernels::prepare(&runtime)?;
        let buffers = ModelTensors::allocate(runtime.clone(), checkpoint, &shape)?;

        let mut worker = Scheme::new(&ctx);
        kernels
            .embed
            .record(
                &mut worker,
                "embed",
                &layout.embedding(&buffers.weights, &shape)?,
                &buffers.step,
                &buffers.x,
                shape.dim,
            )
            .over_tensor(&buffers.x.view());

        for layer in 0..shape.n_layers as usize {
            let views = layout.layer_views(&buffers.weights, layer, &shape)?;
            record_layer(&mut worker, &mut kernels, &buffers, &shape, &views, layer)?;
        }

        let rms_final = layout.rms_final(&buffers.weights, &shape)?;
        kernels
            .rmsnorm_inplace
            .record(
                &mut worker,
                "rmsnorm_final",
                &buffers.x,
                &rms_final,
                shape.dim,
                u32::try_from(rms_final.storage_offset()).unwrap(),
            )
            .groups([1, 1, 1]);

        record_gemv(
            &mut kernels,
            &mut worker,
            "classifier",
            buffers.x.view(),
            layout.classifier(&buffers.weights, &shape)?,
            buffers.logits.view(),
        )?;

        let memory = MemoryExchange::new(&ctx);
        let withdraw = memory.bind_withdraw(&mut worker, buffers.logits.buffer())?;

        let mut upload = Scheme::new(&ctx);
        let deposit = memory.bind_deposit(
            &mut upload,
            DepositTarget::buffer_elements::<DecodeStep>(&buffers.step, 1),
        )?;

        Ok(Self {
            config,
            _tensors: buffers,
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

impl AutoregressiveModel for Model {
    fn vocab_size(&self) -> usize {
        self.config.vocab_size()
    }

    fn max_seq_len(&self) -> u32 {
        self.config.max_seq_len() as u32
    }

    fn step(&mut self, token: u32, pos: u32) -> Result<Vec<f32>> {
        Model::step(self, token, pos)
    }
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn record_gemv(
    kernels: &mut PreparedKernels,
    worker: &mut Scheme,
    label: impl Into<String>,
    x: TensorView<'_>,
    w: TensorView<'_>,
    out: TensorView<'_>,
) -> Result<()> {
    kernels.tensors.matmul_into(worker, label, w, x, out)
}

fn record_layer(
    worker: &mut Scheme,
    kernels: &mut PreparedKernels,
    buffers: &ModelTensors,
    shape: &ModelShape,
    weights: &LayerWeightViews<'_>,
    layer: usize,
) -> Result<()> {
    let loff = layer as u32 * shape.seq_len * shape.kv_dim;
    record_attention_block(worker, kernels, buffers, shape, weights, layer, loff)?;
    record_ffn_block(worker, kernels, buffers, shape, weights, layer)
}

fn record_attention_block(
    worker: &mut Scheme,
    kernels: &mut PreparedKernels,
    buffers: &ModelTensors,
    shape: &ModelShape,
    weights: &LayerWeightViews<'_>,
    layer: usize,
    loff: u32,
) -> Result<()> {
    kernels
        .rmsnorm
        .record(
            worker,
            leak(format!("rmsnorm_att_{layer}")),
            &buffers.x,
            &weights.rms_att,
            &buffers.xb,
            shape.dim,
            u32::try_from(weights.rms_att.storage_offset()).unwrap(),
        )
        .groups([1, 1, 1]);

    record_gemv(
        kernels,
        worker,
        format!("wq_{layer}"),
        buffers.xb.view(),
        weights.wq,
        buffers.q.view(),
    )?;

    kernels
        .gemv
        .record(
            worker,
            leak(format!("wk_{layer}")),
            &buffers.xb,
            &weights.wk,
            &buffers.key_cache,
            &buffers.step,
            shape.dim,
            shape.kv_dim,
            u32::try_from(weights.wk.storage_offset()).unwrap(),
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
            &weights.wv,
            &buffers.value_cache,
            &buffers.step,
            shape.dim,
            shape.kv_dim,
            u32::try_from(weights.wv.storage_offset()).unwrap(),
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
            DEFAULT_ROPE_THETA,
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

    record_gemv(
        kernels,
        worker,
        format!("wo_{layer}"),
        buffers.xb.view(),
        weights.wo,
        buffers.xb2.view(),
    )?;

    kernels.tensors.add_into(
        worker,
        format!("residual_att_{layer}"),
        buffers.x.view(),
        buffers.xb2.view(),
        buffers.x.view(),
    )?;
    Ok(())
}

fn record_ffn_block(
    worker: &mut Scheme,
    kernels: &mut PreparedKernels,
    buffers: &ModelTensors,
    shape: &ModelShape,
    weights: &LayerWeightViews<'_>,
    layer: usize,
) -> Result<()> {
    kernels
        .rmsnorm
        .record(
            worker,
            leak(format!("rmsnorm_ffn_{layer}")),
            &buffers.x,
            &weights.rms_ffn,
            &buffers.xb,
            shape.dim,
            u32::try_from(weights.rms_ffn.storage_offset()).unwrap(),
        )
        .groups([1, 1, 1]);

    record_gemv(
        kernels,
        worker,
        format!("w1_{layer}"),
        buffers.xb.view(),
        weights.w1,
        buffers.hb.view(),
    )?;

    record_gemv(
        kernels,
        worker,
        format!("w3_{layer}"),
        buffers.xb.view(),
        weights.w3,
        buffers.hb2.view(),
    )?;

    kernels
        .swiglu
        .record(
            worker,
            leak(format!("swiglu_{layer}")),
            &buffers.hb,
            &buffers.hb2,
            shape.hidden_dim,
        )
        .over_tensor(&buffers.hb.view());

    record_gemv(
        kernels,
        worker,
        format!("w2_{layer}"),
        buffers.hb.view(),
        weights.w2,
        buffers.xb.view(),
    )?;

    kernels.tensors.add_into(
        worker,
        format!("residual_ffn_{layer}"),
        buffers.x.view(),
        buffers.xb.view(),
        buffers.x.view(),
    )?;
    Ok(())
}
