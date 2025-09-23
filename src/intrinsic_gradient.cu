#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// Each thread computes the gradient block for one face
// Interleaved convention: rows hold vector components as [x0, y0, x1, y1, x2, y2, ...]
template <typename scalar_t, typename index_t>
__global__ void intrinsic_gradient_kernel(
    const scalar_t* __restrict__ edge_lengths, // shape: (F, 3)
    const index_t*  __restrict__ faces,        // shape: (F, 3)
    int64_t* __restrict__ indices,             // shape: (2, 6F)
    scalar_t* __restrict__ values,             // shape: (6F)
    const int64_t F
) {
  const int64_t f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;

  // read the three edge lengths for face f.
  // convention: edge_lengths[f, 0] is the edge opposite vertex 0, etc.
  const scalar_t l0 = edge_lengths[f * 3 + 0];
  const scalar_t l1 = edge_lengths[f * 3 + 1];
  const scalar_t l2 = edge_lengths[f * 3 + 2];

  // Construct a 2D triangle:
  //   p0 = (0, 0)
  //   p1 = (l2, 0)
  //   p2 = (x, y)
  const scalar_t x = (l1 * l1 + l2 * l2 - l0 * l0) / (static_cast<scalar_t>(2) * l2);
  const scalar_t tmp = l1 * l1 - x * x;
  const scalar_t y = sqrt(tmp > 0 ? tmp : 0);
  const scalar_t A = static_cast<scalar_t>(0.5) * l2 * y;  // triangle area
  const scalar_t inv2A = static_cast<scalar_t>(1.0) / (static_cast<scalar_t>(2.0) * A);

  // grad phi0 = perp(p2-p1)/(2A) = (-y, x-l2)/(2A)
  const scalar_t g0_x = -y * inv2A;
  const scalar_t g0_y = (x - l2) * inv2A;

  // grad phi1 = perp(p0-p2)/(2A) = (y, -x)/(2A)
  const scalar_t g1_x = y * inv2A;
  const scalar_t g1_y = -x * inv2A;

  // grad phi2 = perp(p1-p0)/(2A) = (0, l2)/(2A)
  const scalar_t g2_x = static_cast<scalar_t>(0);
  const scalar_t g2_y = l2 * inv2A;

  // vertex indices for face f
  const int64_t v0 = static_cast<int64_t>(faces[f * 3 + 0]);
  const int64_t v1 = static_cast<int64_t>(faces[f * 3 + 1]);
  const int64_t v2 = static_cast<int64_t>(faces[f * 3 + 2]);

  // face contributes 6 entries (2 rows × 3 vertices)
  const int64_t base = f * 6;
  // The output indices tensor has shape (2, 6F):
  // First row: row indices; Second row: column indices.
  // Fill row indices:
  indices[0 * (F * 6) + base + 0] = 2 * f;      // row for g0_x (vertex v0)
  indices[0 * (F * 6) + base + 1] = 2 * f;      // row for g1_x (vertex v1)
  indices[0 * (F * 6) + base + 2] = 2 * f;      // row for g2_x (vertex v2)
  indices[0 * (F * 6) + base + 3] = 2 * f + 1;  // row for g0_y (vertex v0)
  indices[0 * (F * 6) + base + 4] = 2 * f + 1;  // row for g1_y (vertex v1)
  indices[0 * (F * 6) + base + 5] = 2 * f + 1;  // row for g2_y (vertex v2)

  // Fill column indices:
  indices[1 * (F * 6) + base + 0] = v0;  // column for g0_x
  indices[1 * (F * 6) + base + 1] = v1;  // column for g1_x
  indices[1 * (F * 6) + base + 2] = v2;  // column for g2_x
  indices[1 * (F * 6) + base + 3] = v0;  // column for g0_y
  indices[1 * (F * 6) + base + 4] = v1;  // column for g1_y
  indices[1 * (F * 6) + base + 5] = v2;  // column for g2_y

  // Write the corresponding gradient values.
  values[base + 0] = g0_x;
  values[base + 1] = g1_x;
  values[base + 2] = g2_x;
  values[base + 3] = g0_y;
  values[base + 4] = g1_y;
  values[base + 5] = g2_y;
}

void intrinsic_gradient_cuda(
    torch::Tensor edge_lengths,  // (F, 3)
    torch::Tensor faces,         // (F, 3)
    torch::Tensor indices,       // (2, 6F)
    torch::Tensor values         // (6F)
) {
  const int64_t F = edge_lengths.size(0);
  const int threads = 256;
  const int blocks = static_cast<int>((F + threads - 1) / threads);

  c10::cuda::CUDAGuard guard(edge_lengths.device());
  auto stream = at::cuda::getCurrentCUDAStream(edge_lengths.device().index());

  AT_DISPATCH_FLOATING_TYPES(edge_lengths.scalar_type(), "intrinsic_gradient_cuda", [&] {
    using scalar_t_ = scalar_t;
    if (faces.scalar_type() == at::kLong) {
      intrinsic_gradient_kernel<scalar_t_, int64_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          edge_lengths.data_ptr<scalar_t_>(),
          faces.data_ptr<int64_t>(),
          indices.data_ptr<int64_t>(),
          values.data_ptr<scalar_t_>(),
          F);
    } else if (faces.scalar_type() == at::kInt) {
      intrinsic_gradient_kernel<scalar_t_, int32_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          edge_lengths.data_ptr<scalar_t_>(),
          faces.data_ptr<int32_t>(),
          indices.data_ptr<int64_t>(),
          values.data_ptr<scalar_t_>(),
          F);
    } else {
      TORCH_CHECK(false, "faces must be int32 or int64");
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor intrinsic_gradient_operator(
  torch::Tensor edge_lengths, // (F, 3) float32/float64
  torch::Tensor faces         // (F, 3) int64/int32
) {
  TORCH_CHECK(edge_lengths.is_cuda(), "edge_lengths must be a CUDA tensor");
  TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
  TORCH_CHECK(edge_lengths.dim() == 2 && edge_lengths.size(1) == 3,
              "edge_lengths must be of shape (F, 3)");
  TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3,
              "faces must be of shape (F, 3)");

  edge_lengths = edge_lengths.contiguous();
  faces = faces.contiguous();

  const int64_t F = edge_lengths.size(0);
  // compute V from the maximum vertex index over all batches (assumes all batches use the same V)
  const int64_t V = faces.max().item<int64_t>() + 1;

  // allocate indices tensor for sparse matrix
  auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(edge_lengths.device());
  // shape: (2, 6F) where first row holds row indices, second row holds column indices
  torch::Tensor indices = torch::empty({2, F * 6}, options_int);

  // allocate nonzero values (6 per face)
  auto options_float = torch::TensorOptions().dtype(edge_lengths.dtype()).device(edge_lengths.device());
  torch::Tensor values = torch::empty({F * 6}, options_float);

  intrinsic_gradient_cuda(edge_lengths, faces, indices, values);
  std::vector<int64_t> size = {F * 2, V};
  auto grad_operator = torch::sparse_coo_tensor(indices, values, size);
  return grad_operator;
}

void init_intrinsic_gradient(py::module &m) {
    m.def("intrinsic_gradient",
          &intrinsic_gradient_operator,
          "Intrinsic gradient operator (CUDA)");
  }
  
