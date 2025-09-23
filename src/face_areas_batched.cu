#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// Each thread computes the area for one face (in one batch)
template <typename scalar_t, typename index_t>
__global__ void face_areas_batched_kernel(
    const scalar_t* __restrict__ vertices,  // shape: (B, V, 3)
    const index_t*  __restrict__ faces,     // shape: (B, F, 3)
    scalar_t* __restrict__ areas,             // shape: (B, F)
    const int64_t B,
    const int64_t V,
    const int64_t F
) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_faces = B * F;
  if (idx >= total_faces) return;

  // batch index and face index
  const int64_t b = idx / F;
  const int64_t f = idx % F;

  const int64_t face_offset = (b * F + f) * 3;
  index_t v0_idx = faces[face_offset + 0];
  index_t v1_idx = faces[face_offset + 1];
  index_t v2_idx = faces[face_offset + 2];
  const int64_t v0_offset = (static_cast<int64_t>(b) * V + static_cast<int64_t>(v0_idx)) * 3;
  const int64_t v1_offset = (static_cast<int64_t>(b) * V + static_cast<int64_t>(v1_idx)) * 3;
  const int64_t v2_offset = (static_cast<int64_t>(b) * V + static_cast<int64_t>(v2_idx)) * 3;

  // Load vertex coordinates
  scalar_t p0x = vertices[v0_offset + 0];
  scalar_t p0y = vertices[v0_offset + 1];
  scalar_t p0z = vertices[v0_offset + 2];

  scalar_t p1x = vertices[v1_offset + 0];
  scalar_t p1y = vertices[v1_offset + 1];
  scalar_t p1z = vertices[v1_offset + 2];

  scalar_t p2x = vertices[v2_offset + 0];
  scalar_t p2y = vertices[v2_offset + 1];
  scalar_t p2z = vertices[v2_offset + 2];

  // edge vectors: e1 = p1 - p0, e2 = p2 - p0
  scalar_t e1x = p1x - p0x;
  scalar_t e1y = p1y - p0y;
  scalar_t e1z = p1z - p0z;

  scalar_t e2x = p2x - p0x;
  scalar_t e2y = p2y - p0y;
  scalar_t e2z = p2z - p0z;

  // cross product e1 x e2
  scalar_t cx = e1y * e2z - e1z * e2y;
  scalar_t cy = e1z * e2x - e1x * e2z;
  scalar_t cz = e1x * e2y - e1y * e2x;

  // Area = 0.5 * norm(cross)
  scalar_t norm = sqrt(cx * cx + cy * cy + cz * cz);
  areas[b * F + f] = static_cast<scalar_t>(0.5) * norm;
}

void face_areas_batched_cuda_forward(
    torch::Tensor vertices,  // (B, V, 3)
    torch::Tensor faces,     // (B, F, 3)
    torch::Tensor areas      // (B, F)
) {
  const int64_t B = vertices.size(0);
  const int64_t V = vertices.size(1);
  const int64_t F = faces.size(1);
  const int threads = 256;
  const int64_t total_faces = B * F;
  const int blocks = static_cast<int>((total_faces + threads - 1) / threads);

  c10::cuda::CUDAGuard guard(vertices.device());
  auto stream = at::cuda::getCurrentCUDAStream(vertices.device().index());

  AT_DISPATCH_FLOATING_TYPES(vertices.scalar_type(), "face_areas_batched_cuda_forward", [&] {
    using scalar_t_ = scalar_t;
    if (faces.scalar_type() == at::kLong) {
      face_areas_batched_kernel<scalar_t_, int64_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          vertices.data_ptr<scalar_t_>(),
          faces.data_ptr<int64_t>(),
          areas.data_ptr<scalar_t_>(),
          B, V, F);
    } else if (faces.scalar_type() == at::kInt) {
      face_areas_batched_kernel<scalar_t_, int32_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          vertices.data_ptr<scalar_t_>(),
          faces.data_ptr<int32_t>(),
          areas.data_ptr<scalar_t_>(),
          B, V, F);
    } else {
      TORCH_CHECK(false, "faces must be int32 or int64");
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor face_areas_batched(
    torch::Tensor vertices,  // (B, V, 3)
    torch::Tensor faces      // (B, F, 3)
) {
  TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
  TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
  TORCH_CHECK(vertices.dim() == 3 && vertices.size(2) == 3,
              "vertices must be of shape (B, V, 3)");
  TORCH_CHECK(faces.dim() == 3 && faces.size(2) == 3,
              "faces must be of shape (B, F, 3)");

  vertices = vertices.contiguous();
  faces = faces.contiguous();

  const int64_t B = vertices.size(0);
  const int64_t F = faces.size(1);

  auto options = torch::TensorOptions().dtype(vertices.dtype()).device(vertices.device());

  torch::Tensor areas = torch::empty({B, F}, options);
  face_areas_batched_cuda_forward(vertices, faces, areas);
  return areas;
}

void init_face_areas_batched(py::module &m) {
    m.def("face_areas_batched",
          &face_areas_batched,
          "Face areas (CUDA)");
  }
  
