import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  setGaussianScaling: (scale: number) => void;
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  // Splats: center_ndc (vec2), radius_ndc (vec2) => 16 bytes per splat
  const splatStride = 28;
  const splatBuffer = createBuffer(
    device,
    'splats buffer',
    pc.num_points * splatStride,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  );

  // Settings buffer (gaussian_scaling, sh_deg, padding to 16 bytes)
  const settingsBufferSize = 16;
  const settingsInit = new Float32Array([1.0, pc.sh_deg, 0.0, 0.0]);
  const settingsBuffer = createBuffer(
    device,
    'render settings',
    settingsBufferSize,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    settingsInit
  );

  // Indirect draw buffer: [vertexCount, instanceCount, firstVertex, firstInstance]
  // We render a quad as 2 triangles => 6 vertices per instance.
  const indirectDrawBuffer = createBuffer(
    device,
    'draw indirect',
    4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    new Uint32Array([6, pc.num_points, 0, 0]) // overridden after preprocess
  );

  const nulling_data = new Uint32Array([0]);

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  // group(0): camera
  const preprocess_camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });

  // group(1): gaussians (input), splats (output), settings (uniform)
  const preprocess_data_bind_group = device.createBindGroup({
    label: 'preprocess data',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splatBuffer } },
      { binding: 2, resource: { buffer: settingsBuffer } },
      { binding: 3, resource: { buffer: pc.sh_buffer } },
    ],
  });

  // group(2): only binding 0 is present in layout (others are optimized out as unused)
  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  
  const render_shader = device.createShaderModule({ code: renderWGSL });

  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        blend: {
          color: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
        },
      }],
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'none',
    },
  });


  const render_splats_bind_group = device.createBindGroup({
    label: 'render splats',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splatBuffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 2, resource: { buffer: camera_buffer } }
    ],
  });



  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const zero = new Uint32Array([0]);
  const resetCounters = () => {
    device.queue.writeBuffer(sorter.sort_info_buffer, 0, zero);                  // keys_size = 0
    device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, zero);     // dispatch_x = 0
  };
  
  const dispatchPreprocess = (encoder: GPUCommandEncoder) => {
    // reset visible count: sort_infos.keys_size = 0
    resetCounters();
    const pass = encoder.beginComputePass({ label: 'preprocess pass' });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, preprocess_camera_bind_group);
    pass.setBindGroup(1, preprocess_data_bind_group);
    pass.setBindGroup(2, sort_bind_group);

    const wgSize = C.histogram_wg_size;
    const numWG = Math.ceil(pc.num_points / wgSize);
    pass.dispatchWorkgroups(numWG, 1, 1);
    pass.end();


  };

  const recordRender = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        { view: texture_view, loadOp: 'clear', storeOp: 'store' },
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_splats_bind_group);
    pass.drawIndirect(indirectDrawBuffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder, texture_view) => {
      dispatchPreprocess(encoder);

      sorter.sort(encoder); 

      // keys_size → instanceCount
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirectDrawBuffer, 4, 4);

      recordRender(encoder, texture_view);

    },
    camera_buffer,
    setGaussianScaling: (scale: number) => {
      const data = new Float32Array([scale, pc.sh_deg, 0, 0]);
      device.queue.writeBuffer(settingsBuffer, 0, data);
    },
  };
}
