//! Algebraic GPU kernel checks (not a CPU transformer).

#![cfg(any(feature = "cuda", feature = "metal"))]

use goldy::{
    BufferKind, ComputePipeline, DepositTarget, MemoryExchange, NodeAccess, Runtime, Scheme,
    ShaderModule,
};
use llama3_goldy::gpu::create_runtime;
use llama3_goldy::kernels::{AccumKernel, EmbedKernel, MatmulKernel};
use llama3_goldy::shaders;

fn runtime() -> Runtime {
    create_runtime().expect("goldy runtime")
}

fn read_f32(scheme: &mut Scheme, buf: &goldy::Buffer) -> Vec<f32> {
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(scheme, buf)
        .expect("withdraw");
    let mut sub = scheme.submit().expect("submit");
    let bytes = grant
        .claim(&mut sub)
        .expect("claim")
        .consume()
        .expect("consume");
    bytemuck::cast_slice(&bytes).to_vec()
}

#[test]
fn embed_gathers_selected_row() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let embed = device
        .acquire_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0], BufferKind::Scattered)
        .unwrap();
    let control = device
        .acquire_buffer_with_data(&[1u32, 0u32], BufferKind::Scattered)
        .unwrap();
    let x = device
        .acquire_buffer_with_data(&[0.0f32, 0.0], BufferKind::Scattered)
        .unwrap();
    let kernel = EmbedKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(&mut scheme, "embed", &embed, &control, &x, 2)
        .over_1d(2);
    let got = read_f32(&mut scheme, &x);
    assert_eq!(got, vec![3.0, 4.0]);
}

#[test]
fn rope_at_pos_zero_is_identity() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let q = device
        .acquire_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0], BufferKind::Scattered)
        .unwrap();
    let k = device
        .acquire_buffer_with_data(&[5.0f32, 6.0, 7.0, 8.0], BufferKind::Scattered)
        .unwrap();
    let control = device
        .acquire_buffer_with_data(&[0u32, 0u32], BufferKind::Scattered)
        .unwrap();
    let pipeline = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, shaders::ROPE).unwrap(),
    )
    .unwrap();
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("rope", &pipeline)
        .with_parcel(&q, NodeAccess::ReadWrite)
        .with_parcel(&k, NodeAccess::ReadWrite)
        .with_parcel(&control, NodeAccess::Read)
        .with_param(4) // kv_dim
        .with_param(2) // head_size
        .with_param(4) // dim
        .with_param(0) // loff
        .dispatch(1, 1, 1);
    let q_out = read_f32(&mut scheme, &q);
    assert_eq!(q_out, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn matmul_identity_and_accum() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let x = device
        .acquire_buffer_with_data(&[1.0f32, 2.0], BufferKind::Scattered)
        .unwrap();
    let w = device
        .acquire_buffer_with_data(&[1.0f32, 0.0, 0.0, 1.0], BufferKind::Scattered)
        .unwrap();
    let out = device
        .acquire_buffer_with_data(&[0.0f32, 0.0], BufferKind::Scattered)
        .unwrap();
    let control = device
        .acquire_buffer_with_data(&[0u32, 0u32], BufferKind::Scattered)
        .unwrap();
    let matmul = MatmulKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    matmul
        .record(&mut scheme, "mm", &x, &w, &out, &control, 2, 2, 0, 0, 0)
        .over_1d(2);
    assert_eq!(read_f32(&mut scheme, &out), vec![1.0, 2.0]);

    let b = device
        .acquire_buffer_with_data(&[3.0f32, 4.0], BufferKind::Scattered)
        .unwrap();
    let accum = AccumKernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    accum.record(&mut scheme, "acc", &out, &b, 2).over_1d(2);
    assert_eq!(read_f32(&mut scheme, &out), vec![4.0, 6.0]);
}

#[test]
fn silu_of_zero_is_zero() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let hb = device
        .acquire_buffer_with_data(&[0.0f32, 0.0], BufferKind::Scattered)
        .unwrap();
    let hb2 = device
        .acquire_buffer_with_data(&[5.0f32, 7.0], BufferKind::Scattered)
        .unwrap();
    let pipeline = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, shaders::SILU).unwrap(),
    )
    .unwrap();
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("silu", &pipeline)
        .with_parcel(&hb, NodeAccess::ReadWrite)
        .with_parcel(&hb2, NodeAccess::Read)
        .with_param(2)
        .dispatch(1, 1, 1);
    assert_eq!(read_f32(&mut scheme, &hb), vec![0.0, 0.0]);
}

#[test]
fn rmsnorm_matches_llama3_cuda_formula() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let x = device
        .acquire_buffer_with_data(&[1.0f32, 1.0, 1.0, 1.0], BufferKind::Scattered)
        .unwrap();
    let w = device
        .acquire_buffer_with_data(&[1.0f32, 1.0, 1.0, 1.0], BufferKind::Scattered)
        .unwrap();
    let o = device
        .acquire_buffer_with_data(&[0.0f32; 4], BufferKind::Scattered)
        .unwrap();
    let pipeline = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, shaders::RMSNORM).unwrap(),
    )
    .unwrap();
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("rms", &pipeline)
        .with_parcel(&x, NodeAccess::Read)
        .with_parcel(&w, NodeAccess::Read)
        .with_parcel(&o, NodeAccess::Write)
        .with_param(4)
        .with_param(0)
        .dispatch(1, 1, 1);
    let got = read_f32(&mut scheme, &o);
    let ss = 1.0f32 + 1e-5;
    let scale = 1.0 / ss.sqrt();
    for v in got {
        assert!((v - scale).abs() < 1e-5, "{v} vs {scale}");
    }
}

#[test]
fn deposit_feeds_embed_without_rerecord() {
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let embed = device
        .acquire_buffer_with_data(&[10.0f32, 20.0, 30.0, 40.0], BufferKind::Scattered)
        .unwrap();
    let control = device
        .acquire_buffer_with_data(&[0u32, 0u32], BufferKind::Scattered)
        .unwrap();
    let x = device
        .acquire_buffer_with_data(&[0.0f32, 0.0], BufferKind::Scattered)
        .unwrap();
    let kernel = EmbedKernel::prepare(&device).unwrap();
    let mut worker = Scheme::new(&ctx);
    kernel
        .record(&mut worker, "embed", &embed, &control, &x, 2)
        .over_1d(2);
    let grant = MemoryExchange::new(&ctx)
        .bind_withdraw(&mut worker, &x)
        .unwrap();
    let mut upload = Scheme::new(&ctx);
    let deposit = MemoryExchange::new(&ctx)
        .bind_deposit(
            &mut upload,
            DepositTarget::buffer_elements::<u32>(&control, 2),
        )
        .unwrap();
    for token in [0u32, 1u32] {
        deposit.write_data(0, &[token, 0]).unwrap();
        let _ = upload.submit().unwrap();
        let mut sub = worker.submit().unwrap();
        let bytes = grant.claim(&mut sub).unwrap().consume().unwrap();
        let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        if token == 0 {
            assert_eq!(got, vec![10.0, 20.0]);
        } else {
            assert_eq!(got, vec![30.0, 40.0]);
        }
    }
    assert_eq!(worker.replay_stats().records, 1);
}
