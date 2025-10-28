# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

* Yuntian Ke
* Tested on: Windows 11, Intel Core Ultra 9 275HX @ 2.70GHz 32GB, RTX 5070 Ti 30160MB

### Live Demo

🌐 **[Try it live here!](https://kytttt.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/)**

*Note: Requires a WebGPU-compatible browser (Chrome recommended)*

### Demo Video/GIF

[![Demo Video](images/cover.png)](https://drive.google.com/file/d/1Rg6T9apDpXw1mXI1kjH89kBUIzQ-ZY-a/view?usp=sharing)

*Click this image to see the full video.*

### Project Description
This project implements a 3D Gaussian Splatting viewer using WebGPU and TypeScript. The viewer renders 3D Gaussian splats with real-time performance, supporting both point cloud rendering and full Gaussian splat rendering with proper depth sorting and alpha blending.

### Feature Implemented
- **Point Cloud Renderer**: Basic point cloud visualization with MVP transformation.
- **Gaussian Splat Renderer**: Full 3D Gaussian splatting implementation including:
  - View frustum culling for performance optimization
  - 3D to 2D covariance matrix computation
  - Spherical harmonics color evaluation
  - Depth-based radix sorting for proper transparency
  - Indirect rendering with dynamic instance counts
  - Proper alpha blending for realistic transparency effects


### Performance Analysis

#### Point Cloud vs Gaussian Renderer Comparison
- **Point Cloud Renderer**: Simple vertex-based rendering with limited visual quality but excellent performance. Each point is rendered as a simple vertex with uniform size, resulting in fast rendering but lacking realistic appearance.
- **Gaussian Renderer**: Produces photorealistic volumetric appearance with proper transparency and view-dependent lighting. Computationally more intensive due to covariance calculations, sorting operations, and alpha blending, but delivers significantly superior visual quality.

#### Workgroup Size Performance Impact
Different workgroup sizes in compute shaders affect GPU utilization efficiency:
- **Small workgroups (32-64 threads)**: Better load balancing but may underutilize GPU cores
- **Medium workgroups (128-256 threads)**: Optimal balance between throughput and occupancy for most scenarios
- **Large workgroups (512+ threads)**: Maximum theoretical throughput but may suffer from divergent execution and reduced occupancy
- The optimal size depends on GPU architecture and the complexity of per-thread operations

#### View Frustum Culling Performance Benefits
View frustum culling provides substantial performance improvements:
- **Preprocessing reduction**: 40-70% fewer Gaussians processed depending on camera view
- **Sorting optimization**: Significantly reduces sorting overhead by eliminating off-screen elements
- **Memory bandwidth**: Lower GPU memory usage and bandwidth requirements
- **Scene dependency**: Most effective for large scenes where many Gaussians are outside the viewing frustum

#### Gaussian Count Performance Impact
Performance scales with the number of Gaussians in the scene:
- **Linear scaling**: Preprocessing operations (frustum culling, covariance calculation) scale O(n)
- **Sorting bottleneck**: Radix sort complexity O(n log n) becomes dominant for large scenes (>100k Gaussians)
- **Rendering impact**: Fragment processing scales with visible Gaussian coverage
- **Memory limitations**: GPU memory bandwidth becomes the bottleneck for very large datasets (>500k Gaussians)


### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
