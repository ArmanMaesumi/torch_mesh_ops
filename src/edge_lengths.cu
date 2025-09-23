#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// Each thread computes the edge lengths for one face.
template <typename scalar_t, typename index_t>
__global__ void edge_lengths_kernel(
    const scalar_t* __restrict__ vertices, // (V, 3)
    const index_t*  __restrict__ faces,    // (F, 3)
    scalar_t* __restrict__ edge_lengths,   // (F, 3)
    const int64_t F
) {
  const int64_t f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;

  const index_t v0 = faces[f * 3 + 0];
  const index_t v1 = faces[f * 3 + 1];
  const index_t v2 = faces[f * 3 + 2];

  const int64_t v0_offset = static_cast<int64_t>(v0) * 3;
  const int64_t v1_offset = static_cast<int64_t>(v1) * 3;
  const int64_t v2_offset = static_cast<int64_t>(v2) * 3;

  // Load vertex coordinates
  const scalar_t p0x = vertices[v0_offset + 0];
  const scalar_t p0y = vertices[v0_offset + 1];
  const scalar_t p0z = vertices[v0_offset + 2];

  const scalar_t p1x = vertices[v1_offset + 0];
  const scalar_t p1y = vertices[v1_offset + 1];
  const scalar_t p1z = vertices[v1_offset + 2];

  const scalar_t p2x = vertices[v2_offset + 0];
  const scalar_t p2y = vertices[v2_offset + 1];
  const scalar_t p2z = vertices[v2_offset + 2];

  // Compute edge lengths:
  // Edge opposite vertex 0: between vertices v1 and v2
  scalar_t dx = p2x - p1x;
  scalar_t dy = p2y - p1y;
  scalar_t dz = p2z - p1z;
  const scalar_t len0 = sqrt(dx * dx + dy * dy + dz * dz);

  // Edge opposite vertex 1: between vertices v2 and v0
  dx = p0x - p2x;
  dy = p0y - p2y;
  dz = p0z - p2z;
  const scalar_t len1 = sqrt(dx * dx + dy * dy + dz * dz);

  // Edge opposite vertex 2: between vertices v0 and v1
  dx = p1x - p0x;
  dy = p1y - p0y;
  dz = p1z - p0z;
  const scalar_t len2 = sqrt(dx * dx + dy * dy + dz * dz);

  // Write the computed lengths into the output tensor
  const int64_t out_offset = f * 3;
  edge_lengths[out_offset + 0] = len0;
  edge_lengths[out_offset + 1] = len1;
  edge_lengths[out_offset + 2] = len2;
}

void edge_lengths_cuda_forward(
    torch::Tensor vertices,      // (V, 3)
    torch::Tensor faces,         // (F, 3)
    torch::Tensor edge_lengths   // (F, 3)
) {
  const int64_t F = faces.size(0);
  const int threads = 256;
  const int blocks = static_cast<int>((F + threads - 1) / threads);

  c10::cuda::CUDAGuard guard(vertices.device());
  auto stream = at::cuda::getCurrentCUDAStream(vertices.device().index());

  AT_DISPATCH_FLOATING_TYPES(vertices.scalar_type(), "edge_lengths_cuda_forward", [&] {
    using scalar_t_ = scalar_t;
    if (faces.scalar_type() == at::kLong) {
      edge_lengths_kernel<scalar_t_, int64_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          vertices.data_ptr<scalar_t_>(),
          faces.data_ptr<int64_t>(),
          edge_lengths.data_ptr<scalar_t_>(),
          F);
    } else if (faces.scalar_type() == at::kInt) {
      edge_lengths_kernel<scalar_t_, int32_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          vertices.data_ptr<scalar_t_>(),
          faces.data_ptr<int32_t>(),
          edge_lengths.data_ptr<scalar_t_>(),
          F);
    } else {
      TORCH_CHECK(false, "faces must be int32 or int64");
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor edge_lengths(
    torch::Tensor vertices,  // (V, 3)
    torch::Tensor faces      // (F, 3)
) {
  TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
  TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
  TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3,
              "vertices must be of shape (V, 3)");
  TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3,
              "faces must be of shape (F, 3)");

  vertices = vertices.contiguous();
  faces = faces.contiguous();

  const int64_t F = faces.size(0);

  auto options = torch::TensorOptions().dtype(vertices.dtype())
                                         .device(vertices.device());
  torch::Tensor edge_lengths_out = torch::empty({F, 3}, options);

  edge_lengths_cuda_forward(vertices, faces, edge_lengths_out);
  return edge_lengths_out;
}

void init_edge_lengths(py::module &m) {
  m.def("edge_lengths",
        &edge_lengths,
        "Edge lengths (CUDA)");
}

