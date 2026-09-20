//! Shared Slang sources compiled to Goldy compute pipelines.

pub const EMBED: &str = include_str!("../shaders/embed.slang");
pub const RMSNORM: &str = include_str!("../shaders/rmsnorm.slang");
pub const RMSNORM_INPLACE: &str = include_str!("../shaders/rmsnorm_inplace.slang");
pub const MATMUL: &str = include_str!("../shaders/matmul.slang");
pub const ROPE: &str = include_str!("../shaders/rope.slang");
pub const ATTENTION: &str = include_str!("../shaders/attention.slang");
pub const SILU: &str = include_str!("../shaders/silu.slang");
pub const ACCUM: &str = include_str!("../shaders/accum.slang");

pub const WORKGROUP: u32 = 256;

pub fn div_up(n: u32, d: u32) -> u32 {
    n.div_ceil(d)
}
