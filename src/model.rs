//! Retained Goldy transformer matching llama3.cuda `forward()`.
//!
//! Ammon records each decoder block into its own scheme. This crate includes those
//! schemes (`embed`, `layerN/attn`, `layerN/ffn`, `tail`) and binds the `DecodeStep`
//! deposit on the worker root (before include).

use crate::checkpoint::{Checkpoint, Config, LayerWeightViews, ModelShape};
use ammon::blocks::{AttentionSites, Blocks, FfnSites};
use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::AutoregressiveModel;
use anyhow::{Context, Result};
use goldy::{
    BufferKind, Context as GpuContext, DepositTarget, DepositTransaction, GroupId, MemoryExchange,
    ReplayStats, Runtime, Scheme, SchemeLabel, Tensor, TensorDType, TensorShape, TensorView,
};

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
        let n_weights =
            u32::try_from(checkpoint.weights.len()).context("weight count exceeds u32")?;
        let kv_elems = shape.n_layers * shape.seq_len * shape.kv_dim;
        let att_elems = shape.n_heads * shape.seq_len;
        Ok(Self {
            weights: Tensor::from_f32(
                &runtime,
                TensorShape::vector(n_weights),
                &checkpoint.weights,
            )
            .context("upload weights")?,
            x: Tensor::zeros(&runtime, TensorShape::vector(shape.dim), TensorDType::F32)?,
            xb: Tensor::zeros(&runtime, TensorShape::vector(shape.dim), TensorDType::F32)?,
            xb2: Tensor::zeros(&runtime, TensorShape::vector(shape.dim), TensorDType::F32)?,
            hb: Tensor::zeros(
                &runtime,
                TensorShape::vector(shape.hidden_dim),
                TensorDType::F32,
            )?,
            hb2: Tensor::zeros(
                &runtime,
                TensorShape::vector(shape.hidden_dim),
                TensorDType::F32,
            )?,
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
    tensors: ModelTensors,
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

    fn build(runtime: Runtime, ctx: GpuContext, checkpoint: &Checkpoint) -> Result<Self> {
        let config = checkpoint.config;
        let shape = config.shape();
        let layout = checkpoint.layout;
        let blocks = Blocks::prepare(&runtime)?;
        let buffers = ModelTensors::allocate(runtime.clone(), checkpoint, &shape)?;

        let mut worker = Scheme::new(&ctx);
        let deposit = MemoryExchange::new(&ctx).bind_deposit(
            &mut worker,
            DepositTarget::buffer_elements::<DecodeStep>(&buffers.step, 1),
        )?;

        let mut embed = Scheme::new(&ctx);
        blocks.record_embed(
            &mut embed,
            layout.embedding(&buffers.weights, &shape)?,
            &buffers.step,
            buffers.x.view(),
        )?;
        let mut prev = include_group(&mut worker, "embed", &embed, None)?;

        for layer in 0..shape.n_layers as usize {
            let views = layout.layer_views(&buffers.weights, layer, &shape)?;
            prev = include_layer(
                &mut worker,
                &ctx,
                &blocks,
                &buffers,
                &shape,
                &views,
                layer,
                prev,
            )?;
        }

        let mut tail = Scheme::new(&ctx);
        blocks.record_logits(
            &mut tail,
            buffers.x.view(),
            layout.rms_final(&buffers.weights, &shape)?,
            layout.classifier(&buffers.weights, &shape)?,
            buffers.logits.view(),
        )?;
        include_group(&mut worker, "tail", &tail, Some(prev))?;

        Ok(Self {
            config,
            tensors: buffers,
            worker,
            deposit,
        })
    }

    pub fn step(&mut self, token: u32, pos: u32) -> Result<Vec<f32>> {
        (&self.deposit << &DecodeStep {
            token,
            position: pos,
        })?;
        let mut submission = self.worker.submit()?;
        let logits = (&mut submission >> self.tensors.logits.buffer())
            .take::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .to_vec();
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

    fn step(&mut self, token: u32, pos: u32) -> Result<Vec<f32>> {
        Model::step(self, token, pos)
    }
}

fn layer_kv_cache<'a>(
    cache: &'a Tensor,
    layer: usize,
    shape: &ModelShape,
) -> Result<TensorView<'a>> {
    cache
        .view()
        .reshape(&[
            shape.n_layers,
            shape.seq_len,
            shape.n_kv_heads,
            shape.head_size,
        ])
        .and_then(|v| v.narrow(0, layer as u32, 1))
        .and_then(|v| v.reshape(&[shape.seq_len, shape.n_kv_heads, shape.head_size]))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn include_group(
    worker: &mut Scheme,
    label: impl Into<SchemeLabel>,
    child: &Scheme,
    after: Option<GroupId>,
) -> Result<GroupId> {
    let included = worker.include(label, child)?;
    Ok(match after {
        Some(prev) => included.after(prev).finish(),
        None => included.finish(),
    })
}

fn include_layer(
    worker: &mut Scheme,
    ctx: &GpuContext,
    blocks: &Blocks,
    buffers: &ModelTensors,
    shape: &ModelShape,
    weights: &LayerWeightViews<'_>,
    layer: usize,
    prev: GroupId,
) -> Result<GroupId> {
    let mut attn = Scheme::new(ctx);
    blocks.record_attention(
        &mut attn,
        AttentionSites {
            x: buffers.x.view(),
            xb: buffers.xb.view(),
            xb2: buffers.xb2.view(),
            q: buffers.q.view(),
            att: buffers.att.view(),
            key: layer_kv_cache(&buffers.key_cache, layer, shape)?,
            value: layer_kv_cache(&buffers.value_cache, layer, shape)?,
            step: &buffers.step,
            rms: weights.rms_att,
            wq: weights.wq,
            wk: weights.wk,
            wv: weights.wv,
            wo: weights.wo,
        },
    )?;
    let attn = include_group(worker, format!("layer{layer}/attn"), &attn, Some(prev))?;

    let mut ffn = Scheme::new(ctx);
    blocks.record_ffn(
        &mut ffn,
        FfnSites {
            x: buffers.x.view(),
            xb: buffers.xb.view(),
            hb: buffers.hb.view(),
            hb2: buffers.hb2.view(),
            rms: weights.rms_ffn,
            w1: weights.w1,
            w2: weights.w2,
            w3: weights.w3,
        },
    )?;
    include_group(worker, format!("layer{layer}/ffn"), &ffn, Some(attn))
}
