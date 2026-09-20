//! Remaining Slang sources (need Goldy dialect extensions for shared memory / extra math).

pub const RMSNORM: &str = include_str!("../shaders/rmsnorm.slang");
pub const RMSNORM_INPLACE: &str = include_str!("../shaders/rmsnorm_inplace.slang");
pub const ROPE: &str = include_str!("../shaders/rope.slang");
pub const ATTENTION: &str = include_str!("../shaders/attention.slang");
pub const SILU: &str = include_str!("../shaders/silu.slang");

pub const WORKGROUP: u32 = crate::kernels::WORKGROUP;

pub fn div_up(n: u32, d: u32) -> u32 {
    n.div_ceil(d)
}
