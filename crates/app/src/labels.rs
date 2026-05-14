//! SDF glyph atlas + label-layout helpers.
//!
//! Reads `glyph_atlas.png` + `glyph_atlas.json` (produced by
//! `script/gen-glyph-atlas`), uploads the PNG to a GPU texture, and
//! parses the JSON metrics into a per-character lookup table. The
//! realm-label render pass (`crate::passes::realm_labels`) then asks
//! this module to lay each realm name out along its PCA major-axis
//! baseline and emits a vertex buffer of per-glyph quads.
//!
//! The atlas itself stays static — only the per-realm labels rebuild
//! when `set_settlements()` runs.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use crate::gpu::GpuContext;
use crate::settlements::RealmInfo;

// ---- Atlas metrics --------------------------------------------------------
//
// JSON structure produced by `script/gen-glyph-atlas`. We parse only the
// fields we actually consume; ignore unknown keys so the script can grow
// new metadata without breaking the loader.

#[derive(Debug, Deserialize)]
struct AtlasJson {
    atlas_size: u32,
    sdf_spread_em: f32,
    em_per_atlas_px: f32,
    #[allow(dead_code)]
    ascent_em: f32,
    #[allow(dead_code)]
    descent_em: f32,
    #[allow(dead_code)]
    line_gap_em: f32,
    glyphs: HashMap<String, GlyphJson>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct GlyphJson {
    atlas_xywh: [u32; 4],
    size_em: [f32; 2],
    bearing_em: [f32; 2],
    advance_em: f32,
}

/// Per-character metrics, extracted from the atlas JSON. Coordinates are
/// either in atlas pixels (the four `atlas_*` fields) or EM-fractions
/// (everything else); the layout side multiplies EM-fractions by a
/// `font_size_world_km` knob to land in world units.
#[derive(Debug, Clone, Copy)]
pub struct GlyphMetric {
    /// Atlas tile rectangle in pixels: (x, y, w, h). Zero w/h means the
    /// glyph is whitespace — no quad to emit, but `advance_em` still
    /// moves the pen forward.
    pub atlas_xywh: [u32; 4],
    /// Glyph bitmap dimensions in EM units (without SDF spread).
    pub size_em: [f32; 2],
    /// How far to shift the glyph from the pen position to its
    /// top-left corner, in EM units. `bearing_em.1` is positive up
    /// (font convention), so the top of the glyph is at
    /// `pen_y + bearing_y` and the bottom at `pen_y + bearing_y - size_em.1`.
    pub bearing_em: [f32; 2],
    /// Pen advance after this glyph, in EM units.
    pub advance_em: f32,
}

/// Loaded glyph atlas: GPU texture + per-character metrics.
pub struct GlyphAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// Atlas side length in pixels (square). Used to convert glyph
    /// `atlas_xywh` to UVs.
    pub atlas_size: u32,
    /// SDF half-width in EM units. The shader uses this to convert
    /// "EM derivative of UV" into "atlas-pixels per fragment", which
    /// drives the smoothstep band. Without it, antialiasing widens
    /// or narrows depending on zoom.
    pub sdf_spread_em: f32,
    /// EM-fractions per atlas pixel. Equal to `downsample / high_res`
    /// from the bake script. `font_size_world_km / em_per_atlas_px`
    /// gives the world-km-per-atlas-pixel scale.
    #[allow(dead_code)]
    pub em_per_atlas_px: f32,
    /// Per-character lookup. The key is `char`; chars not present
    /// (e.g. control codes, glyphs missing in the font) get rendered
    /// as zero-width whitespace by `layout_label`.
    glyphs: HashMap<char, GlyphMetric>,
}

impl GlyphAtlas {
    /// Build a `GlyphAtlas` from the JSON metrics text + the decoded
    /// PNG bytes. The PNG must already be RGBA8 of the same dimension
    /// as `atlas_size` square (the bake script enforces this).
    pub fn build(
        gpu: &GpuContext,
        json_text: &str,
        png_width: u32,
        png_height: u32,
        rgba_bytes: &[u8],
    ) -> Result<Self, String> {
        let parsed: AtlasJson = serde_json::from_str(json_text)
            .map_err(|e| format!("glyph_atlas.json: {e}"))?;
        if png_width != parsed.atlas_size || png_height != parsed.atlas_size {
            return Err(format!(
                "glyph_atlas.png size {png_width}x{png_height} != json {0}x{0}",
                parsed.atlas_size
            ));
        }

        let mut glyphs = HashMap::with_capacity(parsed.glyphs.len());
        for (k, v) in parsed.glyphs.iter() {
            // JSON keys are 1-character strings. Skip multi-char (shouldn't
            // happen, but defensive).
            let mut cs = k.chars();
            if let (Some(ch), None) = (cs.next(), cs.next()) {
                glyphs.insert(
                    ch,
                    GlyphMetric {
                        atlas_xywh: v.atlas_xywh,
                        size_em: v.size_em,
                        bearing_em: v.bearing_em,
                        advance_em: v.advance_em,
                    },
                );
            }
        }

        // Upload the atlas as Rgba8Unorm. We could use R8Unorm + 1
        // byte/pixel since only the red channel carries the SDF, but
        // Rgba8Unorm matches the existing world-texture upload helper
        // and the size difference (4 MB vs 1 MB at 1024²) is small.
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("glyph_atlas"),
            size: wgpu::Extent3d {
                width: png_width,
                height: png_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(png_width * 4),
                rows_per_image: Some(png_height),
            },
            wgpu::Extent3d {
                width: png_width,
                height: png_height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&Default::default());

        // Linear filtering: SDF fields are smooth functions, so bilinear
        // sampling between texels is exactly what we want.
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("glyph_atlas sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Ok(GlyphAtlas {
            texture,
            view,
            sampler,
            atlas_size: parsed.atlas_size,
            sdf_spread_em: parsed.sdf_spread_em,
            em_per_atlas_px: parsed.em_per_atlas_px,
            glyphs,
        })
    }

    /// Fetch a glyph's metrics, falling back to a 0-width entry if the
    /// character isn't in the atlas. Whitespace and control codes still
    /// move the pen forward by `space_advance_em` so the layout
    /// doesn't collapse.
    pub fn glyph_or_space(&self, ch: char) -> GlyphMetric {
        if let Some(g) = self.glyphs.get(&ch) {
            return *g;
        }
        // Use the space glyph's advance as the fallback so unknown
        // characters look like a regular gap.
        let space_advance = self
            .glyphs
            .get(&' ')
            .map(|g| g.advance_em)
            .unwrap_or(0.25);
        GlyphMetric {
            atlas_xywh: [0, 0, 0, 0],
            size_em: [0.0, 0.0],
            bearing_em: [0.0, 0.0],
            advance_em: space_advance,
        }
    }
}

// ---- Layout output --------------------------------------------------------

/// One quad-vertex, ready for the realm-labels vertex buffer. The
/// shader takes care of the SDF threshold + smoothstep AA; this struct
/// just supplies the per-vertex world position and atlas UV.
///
/// 16 bytes total — small enough that even a few hundred labels with
/// long names stay well under any sane vertex-buffer budget.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct LabelVertex {
    pub world_xz: [f32; 2],
    pub atlas_uv: [f32; 2],
}

// ---- Layout ---------------------------------------------------------------

/// Settings for the realm-label layout. Tweaked once at the call site.
pub struct LayoutSettings {
    /// Fraction of each realm's bbox-long dimension the rendered text
    /// width should aim to fill. The font size is then derived from
    /// `target_w / (advance_em sum)`, giving big realms big labels and
    /// small realms small labels in a single pass. 0.6 reads as
    /// "label spans most of the country, with breathing room on each
    /// side".
    pub target_bbox_fill: f32,
    /// Hard floors / ceilings on the world-km height of one EM. Big
    /// continents would otherwise produce miles-tall labels that
    /// dwarf the terrain; isolated villages would produce sub-km
    /// labels that pixelate at every zoom.
    pub min_font_size_km: f32,
    pub max_font_size_km: f32,
    /// Extra spacing between glyphs as a fraction of EM. Positive
    /// values give the label that "country atlas" widely-tracked
    /// look. 0.15 ≈ a third of the natural advance, which reads as
    /// "small caps spacing" without breaking ligatures.
    pub letter_spacing_em: f32,
    /// Realms whose total `strength` is below this are not labelled.
    /// Filters out single-village or sub-county realms that would
    /// otherwise clutter the map.
    pub min_strength: f32,
}

impl Default for LayoutSettings {
    fn default() -> Self {
        Self {
            // 60% bbox-long fill: leaves a ~20% margin on each end
            // of the realm so the label doesn't overrun borders.
            target_bbox_fill: 0.60,
            // 6 km / 250 km bracket: at the lower bound, a single-
            // city realm in the Swiss seed renders ~6–10 km tall
            // text; at the upper bound, continent-spanning realms
            // get giant text that still doesn't blow out adjacent
            // realms.
            min_font_size_km: 6.0,
            max_font_size_km: 250.0,
            letter_spacing_em: 0.15,
            min_strength: 50.0,
        }
    }
}

/// Lay out the realm names from `infos` into a single shared vertex
/// buffer's worth of triangles. Each glyph contributes 6 vertices
/// (two CCW triangles).
///
/// The label baseline is the realm's PCA major-axis segment. We
/// compute the natural rendered length of the name (sum of advances
/// in world km) and:
///   * If the baseline is shorter than the rendered name, expand
///     the baseline outward from its midpoint to fit. (Single-city
///     realms always hit this path.)
///   * If the baseline is longer, the name centres on the segment
///     midpoint without filling the full length.
///
/// This is a deliberately simple model — closer to a typeset book
/// than a hand-painted atlas. The medial-axis curved-baseline path
/// is left for a future upgrade once the rest of the pipeline lands.
pub fn build_label_vertices(
    atlas: &GlyphAtlas,
    infos: &[RealmInfo],
    settings: &LayoutSettings,
) -> Vec<LabelVertex> {
    let mut verts: Vec<LabelVertex> = Vec::new();

    for info in infos {
        if info.total_strength < settings.min_strength {
            continue;
        }
        if info.name.is_empty() {
            continue;
        }

        // Glyph metrics are EM-fractions; total advance width of the
        // name in EM is independent of the picked font size, so we
        // can compute it before deciding how big to render.
        let chars: Vec<(char, GlyphMetric)> = info
            .name
            .chars()
            .map(|c| (c, atlas.glyph_or_space(c)))
            .collect();
        if chars.is_empty() {
            continue;
        }
        let n = chars.len();
        let total_em: f32 = chars
            .iter()
            .enumerate()
            .map(|(i, (_, g))| {
                g.advance_em
                    + if i + 1 < n {
                        settings.letter_spacing_em
                    } else {
                        0.0
                    }
            })
            .sum();

        // Pick a per-realm font size so the rendered name occupies
        // `target_bbox_fill` of the realm's longest bbox dimension.
        // The bbox is computed from member-city positions, so it
        // grows with realm size; the resulting font size scales
        // smoothly from village (small bbox → small font) to empire
        // (huge bbox → huge font). Floors / ceilings keep extremes
        // legible.
        let bbox_w = (info.bbox.max[0] - info.bbox.min[0]).max(1.0);
        let bbox_h = (info.bbox.max[1] - info.bbox.min[1]).max(1.0);
        let bbox_long = bbox_w.max(bbox_h);
        let target_w = bbox_long * settings.target_bbox_fill;
        let raw_font_size = target_w / total_em.max(0.05);
        let font_size_world_km = raw_font_size.clamp(
            settings.min_font_size_km,
            settings.max_font_size_km,
        );
        let total_world = total_em * font_size_world_km;

        // Determine the baseline direction + length. Single-city
        // realms have start == end; fall back to a horizontal line
        // through the centroid sized to the rendered text.
        let dx = info.baseline_end[0] - info.baseline_start[0];
        let dy = info.baseline_end[1] - info.baseline_start[1];
        let base_len = (dx * dx + dy * dy).sqrt();
        let (dirx, diry) = if base_len > 1e-3 {
            (dx / base_len, dy / base_len)
        } else {
            (1.0, 0.0)
        };

        // Use whichever is longer: the realm's natural baseline or
        // the rendered text width. Centre on the midpoint either way.
        let span = base_len.max(total_world);
        let mid = info.centroid;
        let start = [
            mid[0] - dirx * span * 0.5,
            mid[1] - diry * span * 0.5,
        ];

        // Pen advances along (dirx, diry) by `advance * font_size`
        // world-km per glyph. Centre the text within the span by
        // starting the pen at `start + (span - total_world)/2 * dir`.
        let lead = (span - total_world) * 0.5;
        let mut pen_offset = lead; // distance along baseline in world km

        // Perpendicular direction (rotated 90° CCW) so we can place
        // the glyph above the baseline. World-XZ is right-handed
        // top-down, so "above" the line going in direction `dir` is
        // the left-perpendicular: (-diry, dirx).
        let perpx = -diry;
        let perpy = dirx;

        let inv_atlas = 1.0 / atlas.atlas_size as f32;

        for (_, g) in chars.iter() {
            let glyph_w_world = g.size_em[0] * font_size_world_km;
            let glyph_h_world = g.size_em[1] * font_size_world_km;

            // Advance / 2 - bearing.x to centre glyph on its advance
            // is overkill for a stylised label; the bake's bearings
            // already place the glyph on the baseline correctly. So
            // top-left of the glyph quad in world space is:
            //   start + dir * (pen_offset + bearing_x_world)
            //         + perp * bearing_y_world  (positive bearing_y = up)
            let bx_world = g.bearing_em[0] * font_size_world_km;
            let by_world = g.bearing_em[1] * font_size_world_km;
            let origin_x = start[0] + dirx * (pen_offset + bx_world) + perpx * by_world;
            let origin_y = start[1] + diry * (pen_offset + bx_world) + perpy * by_world;

            // Advance the pen for the next glyph.
            pen_offset +=
                (g.advance_em + settings.letter_spacing_em) * font_size_world_km;

            // Skip zero-area glyphs (whitespace, missing chars).
            let [ax, ay, aw, ah] = g.atlas_xywh;
            if aw == 0 || ah == 0 || glyph_w_world <= 0.0 || glyph_h_world <= 0.0 {
                continue;
            }

            // Four corners of the glyph quad in world space.
            //   tl ─ tr
            //   │ ╲  │
            //   bl ─ br
            // Pen origin is the *top-left* (since bearing_y is the
            // distance from baseline to top of glyph). The body
            // extends `dir` (right) and `-perp` (down).
            let tl = [origin_x, origin_y];
            let tr = [
                origin_x + dirx * glyph_w_world,
                origin_y + diry * glyph_w_world,
            ];
            let bl = [
                origin_x - perpx * glyph_h_world,
                origin_y - perpy * glyph_h_world,
            ];
            let br = [
                tr[0] - perpx * glyph_h_world,
                tr[1] - perpy * glyph_h_world,
            ];

            // Atlas UVs. PNG row 0 sits at top, but wgpu samples in
            // y-down UV space — same convention. So `atlas_xywh` maps
            // straight through.
            let u0 = ax as f32 * inv_atlas;
            let v0 = ay as f32 * inv_atlas;
            let u1 = (ax + aw) as f32 * inv_atlas;
            let v1 = (ay + ah) as f32 * inv_atlas;

            // Two triangles, CCW. (We don't cull, so winding doesn't
            // strictly matter — but keep CCW for sanity.)
            //   tri 1: tl → bl → br
            //   tri 2: tl → br → tr
            verts.push(LabelVertex { world_xz: tl, atlas_uv: [u0, v0] });
            verts.push(LabelVertex { world_xz: bl, atlas_uv: [u0, v1] });
            verts.push(LabelVertex { world_xz: br, atlas_uv: [u1, v1] });

            verts.push(LabelVertex { world_xz: tl, atlas_uv: [u0, v0] });
            verts.push(LabelVertex { world_xz: br, atlas_uv: [u1, v1] });
            verts.push(LabelVertex { world_xz: tr, atlas_uv: [u1, v0] });
        }

        let _ = pen_offset; // silence unused-mut on degenerate-name early returns
    }

    verts
}
