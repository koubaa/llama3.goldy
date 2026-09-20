//! Rust GPU-dialect kernels that lower to `[goldy_compute]` without Goldy dialect extensions.

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

pub use accum::Kernel as AccumKernel;
pub use embed::Kernel as EmbedKernel;
pub use matmul::Kernel as MatmulKernel;
