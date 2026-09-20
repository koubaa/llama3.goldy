//! Retained Goldy transformer matching llama3.cuda `forward()`.

use crate::checkpoint::{Checkpoint, Config, WeightLayout};
use crate::gpu::create_runtime;
use crate::kernels::{
    AccumKernel, AttentionKernel, EmbedKernel, MatmulKernel, RmsnormInplaceKernel, RmsnormKernel, RopeKernel,
    SiluKernel,
};
use anyhow::{Context, Result};
use goldy::{
    Buffer, BufferKind, Context as GpuContext, DepositTarget, DepositTransaction, MemoryExchange, ReplayStats, Runtime,
    Scheme, WithdrawTransaction,
};

struct Pipelines {
    embed: EmbedKernel,
    rmsnorm: RmsnormKernel,
    rmsnorm_inplace: RmsnormInplaceKernel,
    matmul: MatmulKernel,
    rope: RopeKernel,
    attention: AttentionKernel,
    silu: SiluKernel,
    accum: AccumKernel,
}

pub struct Model {
    #[allow(dead_code)]
    runtime: Runtime,
    pub config: Config,
    _layout: WeightLayout,
    _weights: Buffer,
    _x: Buffer,
    _xb: Buffer,
    _xb2: Buffer,
    _hb: Buffer,
    _hb2: Buffer,
    _q: Buffer,
    _att: Buffer,
    _key_cache: Buffer,
    _value_cache: Buffer,
    _control: Buffer,
    _logits: Buffer,
    _pipelines: Pipelines,
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
        let layout = checkpoint.layout;
        let dim = config.dim() as u32;
        let hidden_dim = config.hidden_dim() as u32;
        let kv_dim = config.kv_dim() as u32;
        let n_heads = config.n_heads() as u32;
        let kv_mul = config.kv_mul() as u32;
        let head_size = config.head_size() as u32;
        let seq_len = config.max_seq_len() as u32;
        let vocab = config.vocab_size() as u32;
        let n_layers = config.n_layers();

        let pipelines = Pipelines {
            embed: EmbedKernel::prepare(&runtime).context("prepare embed kernel")?,
            rmsnorm: RmsnormKernel::prepare(&runtime).context("prepare rmsnorm kernel")?,
            rmsnorm_inplace: RmsnormInplaceKernel::prepare(&runtime)
                .context("prepare rmsnorm_inplace kernel")?,
            matmul: MatmulKernel::prepare(&runtime).context("prepare matmul kernel")?,
            rope: RopeKernel::prepare(&runtime).context("prepare rope kernel")?,
            attention: AttentionKernel::prepare(&runtime).context("prepare attention kernel")?,
            silu: SiluKernel::prepare(&runtime).context("prepare silu kernel")?,
            accum: AccumKernel::prepare(&runtime).context("prepare accum kernel")?,
        };

        let zeros = |n: usize| vec![0f32; n];
        let weights = runtime
            .acquire_buffer_with_data(&checkpoint.weights, BufferKind::Scattered)
            .context("upload weights")?;
        let x = runtime.acquire_buffer_with_data(&zeros(config.dim()), BufferKind::Scattered)?;
        let xb = runtime.acquire_buffer_with_data(&zeros(config.dim()), BufferKind::Scattered)?;
        let xb2 = runtime.acquire_buffer_with_data(&zeros(config.dim()), BufferKind::Scattered)?;
        let hb =
            runtime.acquire_buffer_with_data(&zeros(config.hidden_dim()), BufferKind::Scattered)?;
        let hb2 =
            runtime.acquire_buffer_with_data(&zeros(config.hidden_dim()), BufferKind::Scattered)?;
        let q = runtime.acquire_buffer_with_data(&zeros(config.dim()), BufferKind::Scattered)?;
        let att = runtime.acquire_buffer_with_data(
            &zeros(config.n_heads() * config.max_seq_len()),
            BufferKind::Scattered,
        )?;
        let kv_elems = config.n_layers() * config.max_seq_len() * config.kv_dim();
        let key_cache =
            runtime.acquire_buffer_with_data(&zeros(kv_elems), BufferKind::Scattered)?;
        let value_cache =
            runtime.acquire_buffer_with_data(&zeros(kv_elems), BufferKind::Scattered)?;
        let control = runtime.acquire_buffer_with_data(&[0u32, 0u32], BufferKind::Scattered)?;
        let logits =
            runtime.acquire_buffer_with_data(&zeros(config.vocab_size()), BufferKind::Scattered)?;

        let mut worker = Scheme::new(&ctx);
        pipelines
            .embed
            .record(&mut worker, "embed", &weights, &control, &x, dim)
            .over_1d(dim);

        for layer in 0..n_layers {
            record_layer(
                &mut worker,
                &pipelines,
                layer,
                &config,
                &layout,
                &weights,
                &x,
                &xb,
                &xb2,
                &hb,
                &hb2,
                &q,
                &att,
                &key_cache,
                &value_cache,
                &control,
                dim,
                hidden_dim,
                kv_dim,
                n_heads,
                kv_mul,
                head_size,
                seq_len,
            );
        }

        pipelines
            .rmsnorm_inplace
            .record(
                &mut worker,
                "rmsnorm_final",
                &x,
                &weights,
                dim,
                u32::try_from(layout.rms_final_weight).unwrap(),
            )
            .groups([1, 1, 1]);

        pipelines
            .matmul
            .record(
                &mut worker,
                "classifier",
                &x,
                &weights,
                &logits,
                &control,
                dim,
                vocab,
                u32::try_from(layout.wcls).unwrap(),
                0,
                0,
            )
            .over_1d(vocab);

        let memory = MemoryExchange::new(&ctx);
        let withdraw = memory.bind_withdraw(&mut worker, &logits)?;

        let mut upload = Scheme::new(&ctx);
        let deposit = memory.bind_deposit(
            &mut upload,
            DepositTarget::buffer_elements::<u32>(&control, 2),
        )?;

        Ok(Self {
            runtime,
            config,
            _layout: layout,
            _weights: weights,
            _x: x,
            _xb: xb,
            _xb2: xb2,
            _hb: hb,
            _hb2: hb2,
            _q: q,
            _att: att,
            _key_cache: key_cache,
            _value_cache: value_cache,
            _control: control,
            _logits: logits,
            _pipelines: pipelines,
            worker,
            upload,
            deposit,
            withdraw,
        })
    }

    pub fn step(&mut self, token: u32, pos: u32) -> Result<Vec<f32>> {
        self.deposit.write_data(0, &[token, pos])?;
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

#[allow(clippy::too_many_arguments)]
fn record_layer(
    worker: &mut Scheme,
    pipelines: &Pipelines,
    layer: usize,
    config: &Config,
    layout: &WeightLayout,
    weights: &Buffer,
    x: &Buffer,
    xb: &Buffer,
    xb2: &Buffer,
    hb: &Buffer,
    hb2: &Buffer,
    q: &Buffer,
    att: &Buffer,
    key_cache: &Buffer,
    value_cache: &Buffer,
    control: &Buffer,
    dim: u32,
    hidden_dim: u32,
    kv_dim: u32,
    n_heads: u32,
    kv_mul: u32,
    head_size: u32,
    seq_len: u32,
) {
    let dim_us = config.dim();
    let hidden_us = config.hidden_dim();
    let kv_us = config.kv_dim();
    let loff = (layer * config.max_seq_len() * config.kv_dim()) as u32;

    pipelines
        .rmsnorm
        .record(
            worker,
            leak(format!("rmsnorm_att_{layer}")),
            x,
            weights,
            xb,
            dim,
            layout.rms_att_layer(layer, dim_us),
        )
        .groups([1, 1, 1]);

    pipelines
        .matmul
        .record(
            worker,
            leak(format!("wq_{layer}")),
            xb,
            weights,
            q,
            control,
            dim,
            dim,
            layout.wq_layer(layer, dim_us),
            0,
            0,
        )
        .over_1d(dim);

    pipelines
        .matmul
        .record(
            worker,
            leak(format!("wk_{layer}")),
            xb,
            weights,
            key_cache,
            control,
            dim,
            kv_dim,
            layout.wk_layer(layer, dim_us, kv_us),
            loff,
            kv_dim,
        )
        .over_1d(kv_dim);

    pipelines
        .matmul
        .record(
            worker,
            leak(format!("wv_{layer}")),
            xb,
            weights,
            value_cache,
            control,
            dim,
            kv_dim,
            layout.wv_layer(layer, dim_us, kv_us),
            loff,
            kv_dim,
        )
        .over_1d(kv_dim);

    pipelines
        .rope
        .record(
            worker,
            leak(format!("rope_{layer}")),
            q,
            key_cache,
            control,
            kv_dim,
            head_size,
            dim,
            loff,
        )
        .over_1d((dim / 2).max(1));

    pipelines
        .attention
        .record(
            worker,
            leak(format!("attn_{layer}")),
            q,
            att,
            xb,
            key_cache,
            value_cache,
            control,
            kv_dim,
            kv_mul,
            head_size,
            seq_len,
            loff,
        )
        .groups([n_heads, 1, 1]);

    pipelines
        .matmul
        .record(
            worker,
            leak(format!("wo_{layer}")),
            xb,
            weights,
            xb2,
            control,
            dim,
            dim,
            layout.wo_layer(layer, dim_us),
            0,
            0,
        )
        .over_1d(dim);

    pipelines
        .accum
        .record(worker, leak(format!("accum_att_{layer}")), x, xb2, dim)
        .over_1d(dim);

    pipelines
        .rmsnorm
        .record(
            worker,
            leak(format!("rmsnorm_ffn_{layer}")),
            x,
            weights,
            xb,
            dim,
            layout.rms_ffn_layer(layer, dim_us),
        )
        .groups([1, 1, 1]);

    pipelines
        .matmul
        .record(
            worker,
            leak(format!("w1_{layer}")),
            xb,
            weights,
            hb,
            control,
            dim,
            hidden_dim,
            layout.w1_layer(layer, dim_us, hidden_us),
            0,
            0,
        )
        .over_1d(hidden_dim);

    pipelines
        .matmul
        .record(
            worker,
            leak(format!("w3_{layer}")),
            xb,
            weights,
            hb2,
            control,
            dim,
            hidden_dim,
            layout.w3_layer(layer, dim_us, hidden_us),
            0,
            0,
        )
        .over_1d(hidden_dim);

    pipelines
        .silu
        .record(worker, leak(format!("silu_{layer}")), hb, hb2, hidden_dim)
        .over_1d(hidden_dim);

    pipelines
        .matmul
        .record(
            worker,
            leak(format!("w2_{layer}")),
            hb,
            weights,
            xb,
            control,
            hidden_dim,
            dim,
            layout.w2_layer(layer, dim_us, hidden_us),
            0,
            0,
        )
        .over_1d(dim);

    pipelines
        .accum
        .record(worker, leak(format!("accum_ffn_{layer}")), x, xb, dim)
        .over_1d(dim);
}
