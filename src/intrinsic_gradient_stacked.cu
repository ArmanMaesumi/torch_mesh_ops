#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// Each thread computes the gradient block for one face.
// Stacked variant: rows are [0..F-1] for x-components, [F..2F-1] for y-components.
template <typename scalar_t, typename index_t>
__global__ void intrinsic_gradient_kernel_stacked(
    const scalar_t* __restrict__ edge_lengths, // shape: (F, 3)
    const index_t*  __restrict__ faces,        // shape: (F, 3)
    int64_t* __restrict__ indices,             // shape: (2, 6F)
    scalar_t* __restrict__ values,             // shape: (6F)
    const int64_t F
) {
  const int64_t f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;

  // Read the three edge lengths for face f.
  const scalar_t l0 = edge_lengths[f * 3 + 0];
  const scalar_t l1 = edge_lengths[f * 3 + 1];
  const scalar_t l2 = edge_lengths[f * 3 + 2];

  // Build a 2D triangle in intrinsic space
  const scalar_t x = (l1 * l1 + l2 * l2 - l0 * l0) / (static_cast<scalar_t>(2) * l2);
  const scalar_t tmp = l1 * l1 - x * x;
  const scalar_t y = sqrt(tmp > 0 ? tmp : 0);
  const scalar_t A = static_cast<scalar_t>(0.5) * l2 * y;  // triangle area
  const scalar_t inv2A = static_cast<scalar_t>(1.0) / (static_cast<scalar_t>(2.0) * A);

  // Barycentric gradients
  const scalar_t g0_x = -y * inv2A;
  const scalar_t g0_y = (x - l2) * inv2A;
  const scalar_t g1_x =  y * inv2A;
  const scalar_t g1_y = -x * inv2A;
  const scalar_t g2_x = static_cast<scalar_t>(0);
  const scalar_t g2_y = l2 * inv2A;

  // Face vertex indices
  const int64_t v0 = static_cast<int64_t>(faces[f * 3 + 0]);
  const int64_t v1 = static_cast<int64_t>(faces[f * 3 + 1]);
  const int64_t v2 = static_cast<int64_t>(faces[f * 3 + 2]);

  // Each face contributes 6 entries (2 rows × 3 vertices).
  const int64_t base = f * 6;
  const int64_t rows_stride = F * 6;

  // Stacked row indices
  const int64_t row_x = f;      // x rows first
  const int64_t row_y = F + f;  // y rows stacked after all x

  // Fill row indices
  indices[0 * rows_stride + base + 0] = row_x; // g0_x (v0)
  indices[0 * rows_stride + base + 1] = row_x; // g1_x (v1)
  indices[0 * rows_stride + base + 2] = row_x; // g2_x (v2)
  indices[0 * rows_stride + base + 3] = row_y; // g0_y (v0)
  indices[0 * rows_stride + base + 4] = row_y; // g1_y (v1)
  indices[0 * rows_stride + base + 5] = row_y; // g2_y (v2)

  // Fill column indices
  indices[1 * rows_stride + base + 0] = v0;
  indices[1 * rows_stride + base + 1] = v1;
  indices[1 * rows_stride + base + 2] = v2;
  indices[1 * rows_stride + base + 3] = v0;
  indices[1 * rows_stride + base + 4] = v1;
  indices[1 * rows_stride + base + 5] = v2;

  // Values
  values[base + 0] = g0_x;
  values[base + 1] = g1_x;
  values[base + 2] = g2_x;
  values[base + 3] = g0_y;
  values[base + 4] = g1_y;
  values[base + 5] = g2_y;
}

static inline void intrinsic_gradient_cuda_stacked(
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

  AT_DISPATCH_FLOATING_TYPES(edge_lengths.scalar_type(), "intrinsic_gradient_cuda_stacked", [&] {
    using scalar_t_ = scalar_t;
    if (faces.scalar_type() == at::kLong) {
      intrinsic_gradient_kernel_stacked<scalar_t_, int64_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          edge_lengths.data_ptr<scalar_t_>(),
          faces.data_ptr<int64_t>(),
          indices.data_ptr<int64_t>(),
          values.data_ptr<scalar_t_>(),
          F);
    } else if (faces.scalar_type() == at::kInt) {
      intrinsic_gradient_kernel_stacked<scalar_t_, int32_t>
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

// This function computes the intrinsic gradient operator (stacked rows) and returns a sparse tensor.
torch::Tensor intrinsic_gradient_operator_stacked(
  torch::Tensor edge_lengths, // (F, 3)
  torch::Tensor faces         // (F, 3) int32/int64
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
  // Determine number of vertices (V) from the face tensor.
  const int64_t V = faces.max().item<int64_t>() + 1;

  auto options_int = torch::TensorOptions().dtype(torch::kInt64)
                                          .device(edge_lengths.device());
  torch::Tensor indices = torch::empty({2, F * 6}, options_int);

  auto options_float = torch::TensorOptions().dtype(edge_lengths.dtype())
                                            .device(edge_lengths.device());
  torch::Tensor values = torch::empty({F * 6}, options_float);

  intrinsic_gradient_cuda_stacked(edge_lengths, faces, indices, values);

  // Construct and return the sparse tensor with shape (2F, V)
  std::vector<int64_t> size = {F * 2, V};
  auto grad_operator = torch::sparse_coo_tensor(indices, values, size);
  return grad_operator;
}

void init_intrinsic_gradient_stacked(py::module &m) {
  m.def("intrinsic_gradient_stacked",
        &intrinsic_gradient_operator_stacked,
        "Intrinsic gradient operator (stacked rows, CUDA)");
}

