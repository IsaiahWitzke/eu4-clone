//! Per-pass modules. Each module owns the pipeline + bind groups for one
//! shader stage and exposes its own constructor.
//!
//! After the tile-pyramid rewrite the surface area is much smaller than
//! it used to be:
//!   * `realm_field` bakes the per-pixel argmax-realm texture (driven by
//!     `set_settlements`).
//!   * `realm_labels` draws the SDF-glyph country-name overlay.
//!
//! World rendering is now a two-stage pipeline:
//!   * `tile_bake` (once after assets land) writes the 4 LoD atlases.
//!   * `world_mesh` (per frame) rasterises a heightmap-displaced grid
//!     and samples the appropriate atlas + realm-field.

pub mod realm_field;
pub mod realm_labels;
pub mod tile_bake;
pub mod world_mesh;
