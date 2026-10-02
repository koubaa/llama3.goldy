//! Times one retained GEMV submit over a slice of a large flat weights tensor.
//!
//!   gemv-probe TOTAL_MIB ROWS COLS OFFSET_MIB [STEPS]
//!
//! Isolates whether Goldy's GEMV cost depends on the size of the tensor the
//! weights live in (decode binds per-layer views of one flat checkpoint tensor).

#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("build with --features cuda and/or --features metal");
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn main() -> anyhow::Result<()> {
    use ammon::gpu::create_runtime;
    use ammon::Linear;
    use anyhow::Context;
    use goldy::{MemoryExchange, Scheme, Tensor, TensorDType, TensorShape};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, name: &str| -> anyhow::Result<u64> {
        args.get(i)
            .with_context(|| format!("missing {name}"))?
            .parse()
            .with_context(|| format!("bad {name}"))
    };
    let total = (arg(0, "TOTAL_MIB")? << 20) / 4;
    let rows = arg(1, "ROWS")? as u32;
    let cols = arg(2, "COLS")? as u32;
    let offset = (arg(3, "OFFSET_MIB")? << 20) / 4;
    let steps = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(50usize);
    anyhow::ensure!(offset + u64::from(rows) * u64::from(cols) <= total, "slice exceeds tensor");

    let runtime = create_runtime()?;
    let ctx = runtime.create_context()?;
    let host: Vec<f32> = (0..total).map(|i| ((i % 17) as f32) * 1e-3).collect();
    let weights = Tensor::from_f32(&runtime, TensorShape::vector(u32::try_from(total)?), &host)?;
    drop(host);
    let x = Tensor::from_f32(&runtime, TensorShape::vector(cols), &vec![1.0; cols as usize])?;
    let y = Tensor::zeros(&runtime, TensorShape::vector(rows), TensorDType::F32)?;
    let w = weights
        .view()
        .narrow(0, u32::try_from(offset)?, rows * cols)?
        .reshape(&[rows, cols])?;

    let mut scheme = Scheme::new(&ctx);
    let exchange = MemoryExchange::new(&ctx);
    Linear::new(&runtime)?.record(&mut scheme, w, x.view(), y.view())?;
    let sink = exchange.bind_host_sink(&mut scheme, y.buffer())?;

    let mut times = Vec::with_capacity(steps);
    for _ in 0..steps {
        let t = Instant::now();
        let mut submission = scheme.submit()?;
        let claim = &mut submission >> &sink;
        let out = claim.take::<f32>()?;
        std::hint::black_box(&out);
        times.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let warm = &mut times[steps / 2..];
    warm.sort_by(f64::total_cmp);
    let med = warm[warm.len() / 2];
    let gbs = f64::from(rows) * f64::from(cols) * 4.0 / (med * 1e-6) / 1e9;
    println!(
        "tensor {} MiB  gemv {rows}x{cols} at {} MiB  submit+claim median {med:.1} us  ({gbs:.1} GB/s incl. overhead)",
        total * 4 >> 20,
        offset * 4 >> 20
    );
    Ok(())
}
