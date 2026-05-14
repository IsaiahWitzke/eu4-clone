//! Per-pass modules. Each module owns the pipeline + bind groups for one
//! shader stage and exposes a constructor that returns either a `WorldLayer`
//! (for cached world-anchored data) or a custom `Pass` struct (for the image
//! pass, which renders to swapchain).

pub mod base_heightmap;
pub mod detail_noise;
pub mod erosion;
pub mod image;
pub mod realm_field;
pub mod realm_labels;
