//! Retained Goldy transformer matching llama3.cuda `forward()`.
//!
//! Ammon records decoder blocks into named groups (`embed`, `layerN/attn`,
//! `layerN/ffn`, `tail`) on one retained worker. The `DecodeStep` deposit lives on
//! that submitting root.

use crate::checkpoint::{Checkpoint, Config, LayerWeightViews, ModelShape};
use ammon::blocks::{AttentionSites, Blocks, FfnSites};
use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::AutoregressiveModel;
use anyhow::{Context, Result};
use goldy::{
    BufferKind, DepositTarget, DepositTransaction, GoldyError, HostView, MemoryExchange,
    ReplayStats, Runtime, Scheme, Tensor, TensorDType, TensorShape, TensorView,
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
}

impl ModelTensors {
    fn allocate(runtime: &Runtime, checkpoint: &Checkpoint, shape: &ModelShape) -> Result<Self> {
        let n_weights =
            u32::try_from(checkpoint.weights.len()).context("weight count exceeds u32")?;
        Ok(Self {
            weights: Tensor::from_f32(runtime, TensorShape::vector(n_weights), &checkpoint.weights)
                .context("upload weights")?,
            x: Tensor::zeros(runtime, TensorShape::vector(shape.dim), TensorDType::F32)?,
            xb: Tensor::zeros(runtime, TensorShape::vector(shape.dim), TensorDType::F32)?,
            xb2: Tensor::zeros(runtime, TensorShape::vector(shape.dim), TensorDType::F32)?,
            hb: Tensor::zeros(
                runtime,
                TensorShape::vector(shape.hidden_dim),
                TensorDType::F32,
            )?,
            hb2: Tensor::zeros(
                runtime,
                TensorShape::vector(shape.hidden_dim),
                TensorDType::F32,
            )?,
            q: Tensor::zeros(
                runtime,
                TensorShape::from_dims(&[shape.n_heads, shape.head_size])?,
                TensorDType::F32,
            )?,
            att: Tensor::zeros(
                runtime,
                TensorShape::from_dims(&[shape.n_heads, shape.seq_len])?,
                TensorDType::F32,
            )?,
            key_cache: Tensor::zeros(
                runtime,
                TensorShape::from_dims(&[
                    shape.n_layers,
                    shape.seq_len,
                    shape.n_kv_heads,
                    shape.head_size,
                ])?,
                TensorDType::F32,
            )?,
            value_cache: Tensor::zeros(
                runtime,
                TensorShape::from_dims(&[
                    shape.n_layers,
                    shape.seq_len,
                    shape.n_kv_heads,
                    shape.head_size,
                ])?,
                TensorDType::F32,
            )?,
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

    fn build(runtime: Runtime, ctx: goldy::Context, checkpoint: &Checkpoint) -> Result<Self> {
        let config = checkpoint.config;
        let shape = config.shape();
        let layout = checkpoint.layout;
        let blocks = Blocks::prepare(&runtime)?;
        let buffers = ModelTensors::allocate(&runtime, checkpoint, &shape)?;

        let mut worker = Scheme::new(&ctx);
        let deposit = MemoryExchange::new(&ctx).bind_deposit(
            &mut worker,
            DepositTarget::buffer_elements::<DecodeStep>(&buffers.step, 1),
        )?;

        let embedding = layout.embedding(&buffers.weights, &shape)?;
        worker.group("embed", |embed| {
            blocks.record_embed(embed, embedding, &buffers.step, buffers.x.view())
        })?;

        for layer in 0..shape.n_layers as usize {
            let views = layout.layer_views(&buffers.weights, layer, &shape)?;
            record_layer(&mut worker, &blocks, &buffers, &shape, &views, layer)?;
        }

        let rms_final = layout.rms_final(&buffers.weights, &shape)?;
        let classifier = layout.classifier(&buffers.weights, &shape)?;
        worker.group("tail", |tail| {
            blocks.record_logits(
                tail,
                buffers.x.view(),
                rms_final,
                classifier,
                buffers.logits.view(),
            )
        })?;

        Ok(Self {
            config,
            tensors: buffers,
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

fn layer_kv_cache<'a>(
    cache: &'a Tensor,
    layer: usize,
    shape: &ModelShape,
) -> Result<TensorView<'a>, GoldyError> {
    cache
        .view()
        .narrow(0, layer as u32, 1)
        .and_then(|v| v.reshape(&[shape.seq_len, shape.n_kv_heads, shape.head_size]))
}

fn record_layer(
    worker: &mut Scheme,
    blocks: &Blocks,
    buffers: &ModelTensors,
    shape: &ModelShape,
    weights: &LayerWeightViews<'_>,
    layer: usize,
) -> Result<(), GoldyError> {
    let key = layer_kv_cache(&buffers.key_cache, layer, shape)?;
    let value = layer_kv_cache(&buffers.value_cache, layer, shape)?;
    worker.group(format!("layer{layer}/attn"), |attn| {
        blocks.record_attention(
            attn,
            AttentionSites {
                x: buffers.x.view(),
                xb: buffers.xb.view(),
                xb2: buffers.xb2.view(),
                q: buffers.q.view(),
                att: buffers.att.view(),
                key,
                value,
                step: &buffers.step,
                rms: weights.rms_att,
                wq: weights.wq,
                wk: weights.wk,
                wv: weights.wv,
                wo: weights.wo,
            },
        )
    })?;

    worker.group(format!("layer{layer}/ffn"), |ffn| {
        blocks.record_ffn(
            ffn,
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
        )
    })?;
    Ok(())
}
