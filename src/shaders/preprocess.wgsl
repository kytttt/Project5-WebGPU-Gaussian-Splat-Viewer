const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    // Center in NDC and half-size (radius) in NDC, used for quad reconstruction.
    center_ndc: u32,
    radius_ndc: u32,

    color_rg: u32,
    color_ba: u32,

    conic_xy: u32,
    conic_z_pad: u32,

    opacity_pad: u32,
};

// group(0): camera
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

// group(1): input gaussians, output splats, settings
@group(1) @binding(0)
var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats : array<Splat>;
@group(1) @binding(2)
var<uniform> settings: RenderSettings;
@group(1) @binding(3)
var<storage, read> sh_data : array<u32>;

// group(2): sorting-related buffers (we only use keys_size for visible count in this stage)
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;


fn read_half(i: u32) -> f32 {
    let word_idx: u32 = i >> 1u;           // divide by 2
    let hi: bool = (i & 1u) == 1u;         // odd => high half
    let packed: vec2<f32> = unpack2x16float(sh_data[word_idx]);
    return select(packed.x, packed.y, hi);
}
/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {

    let base = splat_idx * 24u + (c_idx >> 1u) * 3u + (c_idx & 1u);
    let color01 = unpack2x16float(sh_data[base + 0u]);
    let color23 = unpack2x16float(sh_data[base + 1u]);
    
    if (c_idx & 1u) == 0u {
        return vec3f(color01.x, color01.y, color23.x);
    }
    return vec3f(color01.y, color23.x, color23.y);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

// --- Helpers for covariance and projection ---

fn quat_to_mat3(qin: vec4<f32>) -> mat3x3<f32> {
    var q = normalize(qin);
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;

    // Column-major construction
    return mat3x3<f32>(
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
        2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy),
    );
}


fn covariance3d(rot_q: vec4<f32>, scale: vec3<f32>, scale_factor: f32) -> mat3x3<f32> {
    let R = quat_to_mat3(rot_q);
    let s_lin = scale * scale_factor;
    let S = mat3x3<f32>(
        s_lin.x, 0.0, 0.0,
        0.0, s_lin.y, 0.0,
        0.0, 0.0, s_lin.z,
    );

    return transpose(S * R) * (S * R);
}

fn view_rotation_R(view: mat4x4<f32>) -> mat3x3<f32> {

    let Rt = mat3x3<f32>(view[0].xyz, view[1].xyz, view[2].xyz);
    return transpose(Rt);
}

fn jacobian_camera_to_pixel(pos_cam: vec3<f32>, focal: vec2<f32>) -> mat3x3<f32> {
    let x = pos_cam.x;
    let y = pos_cam.y;
    let z = pos_cam.z;
    let fx = focal.x;
    let fy = focal.y;

    return mat3x3<f32>(
        fx / z, 0.0,     -fx * x / (z * z),
        0.0,    fy / z,  -fy * y / (z * z),
        0.0,    0.0,      0.0,
    );
}


fn largest_eigenvalue_2x2(a: f32, b: f32, c: f32) -> f32 {

    let tr = a + c;
    let det = a * c - b * b;
    let disc = max(0.0, tr * tr * 0.25 - det);
    let root = sqrt(disc);
    let l1 = tr * 0.5 + root;
    let l2 = tr * 0.5 - root;
    return max(l1, l2);
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    let count = arrayLength(&gaussians);
    if (idx >= count) {
        return;
    }

    let g = gaussians[idx];

    let a = unpack2x16float(g.pos_opacity[0]);
    let b = unpack2x16float(g.pos_opacity[1]);
    let pos_world = vec4<f32>(a.x, a.y, b.x, 1.0);
    let opacity = b.y;

    let M = camera.proj * camera.view;
    let clip = M * pos_world;


    let depthDetect = (camera.view * pos_world).z;
    if(depthDetect < 0.f) {
        return;
    }

    let ndc = clip.xy / clip.w;


    if (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2) {
        return;
    }

   
    let r01 = unpack2x16float(g.rot[0]);
    let r23 = unpack2x16float(g.rot[1]);
    let rot_q = vec4<f32>(r01.y, r23.x, r23.y, r01.x);

    let s01 = unpack2x16float(g.scale[0]);
    let s23 = unpack2x16float(g.scale[1]);
    let scale = vec3<f32>(exp(s01.x), exp(s01.y), exp(s23.x));

    var t = (camera.view * pos_world).xyz;

    let Sigma3D = covariance3d(rot_q, scale, settings.gaussian_scaling);

    let Vrk = mat3x3<f32>(
        Sigma3D[0][0], Sigma3D[0][1], Sigma3D[0][2],
        Sigma3D[0][1], Sigma3D[1][1], Sigma3D[1][2],
        Sigma3D[0][2], Sigma3D[1][2], Sigma3D[2][2],
    );

    let W = view_rotation_R(camera.view);

    let J = jacobian_camera_to_pixel(t, camera.focal);

    let T = W * J;

    var cov2D = transpose(T) * Vrk * T;
    cov2D[0][0] += 0.3;
    cov2D[1][1] += 0.3;

    let a1 = cov2D[0][0];
    let b1 = cov2D[0][1];
    let c1 = cov2D[1][1];

    let det = a1 * c1 - b1 * b1;
    if (det == 0.0) {
        return;
    }
    let mid = 0.5 * (a1 + c1);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));

    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    let width  = camera.viewport.x;
    let height = camera.viewport.y;

    let rx_ndc = radius * (2.0 / width);
    let ry_ndc = radius * (2.0 / height);

    let packed_center = pack2x16float(ndc);
    let packed_radius = pack2x16float(vec2<f32>(rx_ndc, ry_ndc));

    let write_idx = atomicAdd(&sort_infos.keys_size, 1u);

    splats[write_idx].center_ndc = packed_center;
    splats[write_idx].radius_ndc = packed_radius;

    let view_dir = normalize(pos_world.xyz - camera.view_inv[3].xyz);
    let color = computeColorFromSH(view_dir, idx, u32(settings.sh_deg));

    let packed_rg = pack2x16float(color.xy);
    let packed_ba = pack2x16float(vec2<f32>(color.z, 1.0));

    splats[write_idx].color_rg = packed_rg;
    splats[write_idx].color_ba = packed_ba;

    let det_inv = 1.0 / det;
    let conic = vec3<f32>(c1 * det_inv, -b1 * det_inv, a1 * det_inv);

    let packed_conic_xy = pack2x16float(conic.xy);
    let packed_conic_z_pad = pack2x16float(vec2<f32>(conic.z, 0.0));

    splats[write_idx].conic_xy = packed_conic_xy;
    splats[write_idx].conic_z_pad = packed_conic_z_pad;

    let opacity_sigmoid = sigmoid(opacity);

    let packed_opacity_pad = pack2x16float(vec2<f32>(opacity_sigmoid, 0.0));
    splats[write_idx].opacity_pad = packed_opacity_pad;


    let depth_positive = -depthDetect;
    let depth_bits = bitcast<u32>(depth_positive);
    let sort_key = 0xFFFFFFFFu - depth_bits;
    sort_depths[write_idx] = sort_key; 
    sort_indices[write_idx] = write_idx;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 

    if ((write_idx % keys_per_dispatch) == 0u) {
        _ = atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}