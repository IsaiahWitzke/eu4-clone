//! Spinner overlay pass — a small rotating arc in the top-right corner of
//! the swapchain. Diagnostic only: combined with the continuous-rAF loop
//! in `lib.rs`, smooth rotation means `frame()` is keeping up with the
//! display refresh, while stutter means it isn't.
//!
//! The shader is a fullscreen triangle that `discard`s everything outside
//! a ~26-pixel ring at the corner, so per-frame cost is dominated by the
//! ring's pixel count (a few hundred fragments), not the screen size.
//!
//! Reads:
//!   * SpinnerUniforms (binding 0): elapsed seconds + framebuffer dims.
//!
//! Writes:
//!   * the swapchain (LoadOp::Load) with premultiplied-alpha blending.

use bytemuck::{Pod, Zeroable};

use crate::gpu::GpuContext;

const SHADER_WGSL: &str = include_str!("../shaders/spinner.wgsl");

/// Mirrors `SpinnerUniforms` in `spinner.wgsl`. 16 bytes, std140-friendly.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SpinnerUniforms {
    time_s: f32,
    res_x: f32,
    res_y: f32,
    _pad: f32,
}

pub struct SpinnerPass {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
}

pub fn build(gpu: &GpuContext, target_format: wgpu::TextureFormat) -> SpinnerPass {
    let device = &gpu.device;

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("spinner bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            // Vertex stage doesn't actually read it (vs is parameter-free),
            // but VERTEX_FRAGMENT visibility keeps the binding ergonomic
            // for future tweaks without rebuilding the BGL.
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("spinner"),
        source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("spinner pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    // Premultiplied alpha — same setup as `realm_labels`. The fragment
    // shader emits `(rgb * alpha, alpha)`, so SrcOne + InvSrcA gives the
    // standard "draw on top of the swapchain" behaviour.
    let blend = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("spinner pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spinner ub"),
        size: std::mem::size_of::<SpinnerUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("spinner bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    SpinnerPass {
        pipeline,
        bind_group,
        uniform_buffer,
    }
}

impl SpinnerPass {
    /// Push the latest elapsed time + framebuffer size into the uniform
    /// buffer. Call once per frame before [`render`].
    pub fn write_uniforms(&self, gpu: &GpuContext, time_s: f32, width: u32, height: u32) {
        let u = SpinnerUniforms {
            time_s,
            res_x: width as f32,
            res_y: height as f32,
            _pad: 0.0,
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&u));
    }

    /// Draw the spinner on top of `target_view`. LoadOp::Load preserves
    /// the image-pass output underneath.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("spinner"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
