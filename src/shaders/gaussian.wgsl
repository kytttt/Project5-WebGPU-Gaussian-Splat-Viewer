struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) center_ndc: vec2<f32>,
    @location(2) radius_ndc: vec2<f32>,
    @location(3) conic: vec3<f32>,
    @location(4) opacity: f32,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    center_ndc: u32,
    radius_ndc: u32,
    color_rg: u32,
    color_ba: u32,

    conic_xy: u32,
    conic_z_pad: u32,
    opacity_pad: u32,

};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
};


@group(0) @binding(0)
var<storage, read> splats : array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

// Map vertex_index (0..5) to a triangle-list quad in clip-space
fn corner(ix: u32) -> vec2<f32> {

    switch ix {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>( 1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0, -1.0); }
        case 4u: { return vec2<f32>( 1.0,  1.0); }
        default: { return vec2<f32>(-1.0,  1.0); } 
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let index = sort_indices[iid];
    let s = splats[index];

    let center = unpack2x16float(s.center_ndc);
    let radius = unpack2x16float(s.radius_ndc);

    let color_rg = unpack2x16float(s.color_rg);
    let color_ba = unpack2x16float(s.color_ba);
    let color = vec4<f32>(color_rg.x, color_rg.y, color_ba.x, color_ba.y);  

    let conic_xy = unpack2x16float(s.conic_xy);
    let conic_z = unpack2x16float(s.conic_z_pad).x;
    let conic = vec3<f32>(conic_xy.x, conic_xy.y, conic_z);

    let opacity = unpack2x16float(s.opacity_pad).x;

    let offset = corner(vid) * radius;
    let ndc = vec4<f32>(center + offset, 0.0, 1.0);
    
    out.position = ndc;

    out.color = color;
    out.center_ndc = center;
    out.radius_ndc = radius;
    out.conic = conic;
    out.opacity = opacity;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    var pos_ndc = (in.position.xy / camera.viewport) * 2.0 - 1.0;
    pos_ndc.y = -pos_ndc.y;

    var to_center = pos_ndc - in.center_ndc;
    to_center *= camera.viewport * 0.5;

    let power = -0.5 * (in.conic.x * to_center.x * to_center.x +
                       in.conic.z * to_center.y * to_center.y -
                       2.0 * in.conic.y * to_center.x * to_center.y);

    if (power > 0.0){
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return in.color * min(in.opacity * exp(power), 0.99);
}