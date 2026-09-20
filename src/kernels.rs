//! Rust GPU-dialect kernels matching the llama3.cuda FP32 forward pass.

pub const WORKGROUP: u32 = 256;

/// Gather one embedding row: `x[i] = embed[token * dim + i]`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn embed(embed: &[f32], control: &[u32], x: goldy::gpu::Scattered<f32>, dim: u32) {
    let i = goldy::gpu::global_id().x;
    if i < dim {
        let token = control[0];
        x[i] = embed[token * dim + i];
    }
}

/// Row-major GEMV with optional pos-strided output: `xout[out_base + pos * stride + i]`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn matmul(
    x: &[f32],
    w: &[f32],
    xout: goldy::gpu::Scattered<f32>,
    control: &[u32],
    n: u32,
    d: u32,
    w_offset: u32,
    out_base: u32,
    stride_from_pos: u32,
) {
    let i = goldy::gpu::global_id().x;
    if i < d {
        let pos = control[1];
        let out_i = out_base + pos * stride_from_pos + i;
        let mut sum = 0.0;
        for j in 0..n {
            sum = sum + w[w_offset + i * n + j] * x[j];
        }
        xout[out_i] = sum;
    }
}

/// Residual add: `a[i] += b[i]`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn accum(a: &mut [f32], b: &[f32], size: u32) {
    let i = goldy::gpu::global_id().x;
    if i < size {
        a[i] = a[i] + b[i];
    }
}

/// RMSNorm into `o` (one workgroup, strided over `size`).
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn rmsnorm(
    x: &[f32],
    weight: &[f32],
    o: goldy::gpu::Scattered<f32>,
    size: u32,
    weight_offset: u32,
) {
    let mut scratch = goldy::gpu::workgroup_array::<f32, 256>();
    let local = goldy::gpu::local_id().x;
    let mut ss = 0.0;
    let mut j = local;
    while j < size {
        let v = x[j];
        ss = ss + v * v;
        j = j + 256;
    }
    scratch[local] = ss;
    for i in 0..8 {
        goldy::gpu::workgroup_barrier();
        if local + (1u32 << i) < 256 {
            ss = ss + scratch[local + (1u32 << i)];
        }
        goldy::gpu::workgroup_barrier();
        scratch[local] = ss;
    }
    goldy::gpu::workgroup_barrier();
    ss = scratch[0];
    ss = ss / (size as f32);
    ss = ss + 1e-5;
    ss = 1.0 / goldy::gpu::sqrt(ss);
    j = local;
    while j < size {
        o[j] = weight[weight_offset + j] * (ss * x[j]);
        j = j + 256;
    }
}

/// In-place RMSNorm on `x`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], size: u32, weight_offset: u32) {
    let mut scratch = goldy::gpu::workgroup_array::<f32, 256>();
    let local = goldy::gpu::local_id().x;
    let mut ss = 0.0;
    let mut j = local;
    while j < size {
        let v = x[j];
        ss = ss + v * v;
        j = j + 256;
    }
    scratch[local] = ss;
    for i in 0..8 {
        goldy::gpu::workgroup_barrier();
        if local + (1u32 << i) < 256 {
            ss = ss + scratch[local + (1u32 << i)];
        }
        goldy::gpu::workgroup_barrier();
        scratch[local] = ss;
    }
    goldy::gpu::workgroup_barrier();
    ss = scratch[0];
    ss = ss / (size as f32);
    ss = ss + 1e-5;
    ss = 1.0 / goldy::gpu::sqrt(ss);
    j = local;
    while j < size {
        x[j] = weight[weight_offset + j] * (ss * x[j]);
        j = j + 256;
    }
}

/// Pairwise RoPE on Q and the current K cache row.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn rope(
    q: &mut [f32],
    k: &mut [f32],
    control: &[u32],
    kv_dim: u32,
    head_size: u32,
    dim: u32,
    loff: u32,
) {
    let i = goldy::gpu::global_id().x * 2;
    if i >= dim {
        return;
    }
    let pos = control[1];
    let k_base = loff + pos * kv_dim;
    let head_dim = (i % head_size) as i32;
    let freq = 1.0 / goldy::gpu::pow(10000.0, (head_dim as f32) / (head_size as f32));
    let val = (pos as f32) * freq;
    let fcr = goldy::gpu::cos(val);
    let fci = goldy::gpu::sin(val);
    let mut rotn = 1;
    if i < kv_dim {
        rotn = 2;
    }
    for v in 0..rotn {
        if v == 0 {
            let v0 = q[i];
            let v1 = q[i + 1];
            q[i] = v0 * fcr - v1 * fci;
            q[i + 1] = v0 * fci + v1 * fcr;
        } else {
            let v0 = k[k_base + i];
            let v1 = k[k_base + i + 1];
            k[k_base + i] = v0 * fcr - v1 * fci;
            k[k_base + i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

/// Multi-head attention (one workgroup per head). Inclusive over `t <= pos`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn attention(
    q: &[f32],
    att: &mut [f32],
    xb: goldy::gpu::Scattered<f32>,
    key_cache: &[f32],
    value_cache: &[f32],
    control: &[u32],
    kv_dim: u32,
    kv_mul: u32,
    head_size: u32,
    seq_len: u32,
    loff: u32,
) {
    let mut scratch = goldy::gpu::workgroup_array::<f32, 256>();
    let h = goldy::gpu::workgroup_id().x;
    let local = goldy::gpu::local_id().x;
    let pos = control[1];
    let q_base = h * head_size;
    let att_base = h * seq_len;
    let kv_head = h / kv_mul;

    let mut local_max = -1e30;
    let mut t = local;
    while t <= pos {
        let mut score = 0.0;
        let k_base = loff + t * kv_dim + kv_head * head_size;
        for i in 0..head_size {
            score = score + q[q_base + i] * key_cache[k_base + i];
        }
        score = score / goldy::gpu::sqrt(head_size as f32);
        att[att_base + t] = score;
        if score > local_max {
            local_max = score;
        }
        t = t + 256;
    }

    let mut val = local_max;
    scratch[local] = val;
    for i in 0..8 {
        goldy::gpu::workgroup_barrier();
        if local + (1u32 << i) < 256 {
            val = goldy::gpu::max(val, scratch[local + (1u32 << i)]);
        }
        goldy::gpu::workgroup_barrier();
        scratch[local] = val;
    }
    goldy::gpu::workgroup_barrier();
    let max_val = scratch[0];

    let mut local_sum = 0.0;
    t = local;
    while t <= pos {
        let e = goldy::gpu::exp(att[att_base + t] - max_val);
        att[att_base + t] = e;
        local_sum = local_sum + e;
        t = t + 256;
    }
    val = local_sum;
    scratch[local] = val;
    for i in 0..8 {
        goldy::gpu::workgroup_barrier();
        if local + (1u32 << i) < 256 {
            val = val + scratch[local + (1u32 << i)];
        }
        goldy::gpu::workgroup_barrier();
        scratch[local] = val;
    }
    goldy::gpu::workgroup_barrier();
    let sum = scratch[0];
    t = local;
    while t <= pos {
        att[att_base + t] = att[att_base + t] / sum;
        t = t + 256;
    }
    goldy::gpu::workgroup_barrier();

    let mut i = local;
    while i < head_size {
        let mut acc = 0.0;
        for t in 0..(pos + 1) {
            let v_base = loff + t * kv_dim + kv_head * head_size;
            acc = acc + att[att_base + t] * value_cache[v_base + i];
        }
        xb[q_base + i] = acc;
        i = i + 256;
    }
}

/// SwiGLU: `hb[i] *= silu(hb[i]) * hb2[i]`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn silu(hb: &mut [f32], hb2: &[f32], hidden_dim: u32) {
    let i = goldy::gpu::global_id().x;
    if i >= hidden_dim {
        return;
    }
    let mut val = hb[i];
    val = val * (1.0 / (1.0 + goldy::gpu::exp(-val)));
    val = val * hb2[i];
    hb[i] = val;
}

pub use accum::Kernel as AccumKernel;
pub use attention::Kernel as AttentionKernel;
pub use embed::Kernel as EmbedKernel;
pub use matmul::Kernel as MatmulKernel;
pub use rmsnorm::Kernel as RmsnormKernel;
pub use rmsnorm_inplace::Kernel as RmsnormInplaceKernel;
pub use rope::Kernel as RopeKernel;
pub use silu::Kernel as SiluKernel;
