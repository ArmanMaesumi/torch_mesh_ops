#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// Each thread computes the area for one face
template <typename scalar_t, typename index_t>
__global__ void face_areas_kernel(
    const scalar_t* __restrict__ vertices,  // (V, 3)
    const index_t*  __restrict__ faces,     // (F, 3)
    scalar_t* __restrict__ areas,           // (F)
    const int64_t F
) {
  const int64_t f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;

  const index_t v0 = faces[f * 3 + 0];
  const index_t v1 = faces[f * 3 + 1];
  const index_t v2 = faces[f * 3 + 2];

  const int64_t v0o = static_cast<int64_t>(v0) * 3;
  const int64_t v1o = static_cast<int64_t>(v1) * 3;
  const int64_t v2o = static_cast<int64_t>(v2) * 3;

  // Load vertices
  const scalar_t p0x = vertices[v0o + 0];
  const scalar_t p0y = vertices[v0o + 1];
  const scalar_t p0z = vertices[v0o + 2];

  const scalar_t p1x = vertices[v1o + 0];
  const scalar_t p1y = vertices[v1o + 1];
  const scalar_t p1z = vertices[v1o + 2];

  const scalar_t p2x = vertices[v2o + 0];
  const scalar_t p2y = vertices[v2o + 1];
  const scalar_t p2z = vertices[v2o + 2];

  // edge vectors: e1 = p1 - p0, e2 = p2 - p0
  const scalar_t e1x = p1x - p0x;
  const scalar_t e1y = p1y - p0y;
  const scalar_t e1z = p1z - p0z;

  const scalar_t e2x = p2x - p0x;
  const scalar_t e2y = p2y - p0y;
  const scalar_t e2z = p2z - p0z;

  // cross product e1 x e2
  const scalar_t cx = e1y * e2z - e1z * e2y;
  const scalar_t cy = e1z * e2x - e1x * e2z;
  const scalar_t cz = e1x * e2y - e1y * e2x;

  // Area = 0.5 * norm(cross)
  const scalar_t norm = sqrt(cx * cx + cy * cy + cz * cz);
  areas[f] = static_cast<scalar_t>(0.5) * norm;
}

void face_areas_cuda_forward(
    torch::Tensor vertices,  // (V, 3)
    torch::Tensor faces,     // (F, 3)
    torch::Tensor areas      // (F)
) {
  const int64_t F = faces.size(0);
  const int threads = 256;
  const int blocks = static_cast<int>((F + threads - 1) / threads);

  c10::cuda::CUDAGuard guard(vertices.device());
  auto stream = at::cuda::getCurrentCUDAStream(vertices.device().index());

  AT_DISPATCH_FLOATING_TYPES(vertices.scalar_type(), "face_areas_cuda_forward", [&] {
    using scalar_t_ = scalar_t;
    if (faces.scalar_type() == at::kLong) {
      face_areas_kernel<scalar_t_, int64_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          vertices.data_ptr<scalar_t_>(),
          faces.data_ptr<int64_t>(),
          areas.data_ptr<scalar_t_>(),
          F);
    } else if (faces.scalar_type() == at::kInt) {
      face_areas_kernel<scalar_t_, int32_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          vertices.data_ptr<scalar_t_>(),
          faces.data_ptr<int32_t>(),
          areas.data_ptr<scalar_t_>(),
          F);
    } else {
      TORCH_CHECK(false, "faces must be int32 or int64");
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor face_areas(
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
  auto options = torch::TensorOptions().dtype(vertices.dtype()).device(vertices.device());
  torch::Tensor areas = torch::empty({F}, options);
  face_areas_cuda_forward(vertices, faces, areas);
  return areas;
}

void init_face_areas(py::module &m) {
  m.def("face_areas",
        &face_areas,
        "Face areas (CUDA)");
}

