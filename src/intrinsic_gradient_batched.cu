#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// Each thread processes one face from the entire batch
// Interleaved convention: rows hold vector components as [x0, y0, x1, y1, x2, y2, ...]
template <typename scalar_t, typename index_t>
__global__ void intrinsic_gradient_batched_kernel(
    const scalar_t* __restrict__ edge_lengths, // shape: (B, F, 3)
    const index_t*  __restrict__ faces,        // shape: (B, F, 3)
    int64_t* __restrict__ indices,             // shape: (3, B*6F)
    scalar_t* __restrict__ values,             // shape: (B*6F)
    const int64_t B,
    const int64_t F
) {
  const int64_t face_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_faces = B * F;
  if (face_idx >= total_faces) return;

  // batch index and face index
  const int64_t b = face_idx / F;
  const int64_t f = face_idx % F;

  // offsets for this face in the flattened (B, F, 3) arrays
  const int64_t base_edge = (b * F + f) * 3;
  const int64_t base_face = (b * F + f) * 3;

  // read three edge lengths for this face
  // convention: edge_lengths[b, f, 0] is the edge opposite vertex 0, etc
  const scalar_t l0 = edge_lengths[base_edge + 0];
  const scalar_t l1 = edge_lengths[base_edge + 1];
  const scalar_t l2 = edge_lengths[base_edge + 2];

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

  // vertex indices for this face
  const int64_t v0 = static_cast<int64_t>(faces[base_face + 0]);
  const int64_t v1 = static_cast<int64_t>(faces[base_face + 1]);
  const int64_t v2 = static_cast<int64_t>(faces[base_face + 2]);

  // Each face contributes 6 nonzero entries
  // For each face, we write two rows: row 2*f (for x-components) and row 2*f+1 (for y-components)
  const int64_t base = face_idx * 6;  // base index for this face’s 6 entries in the flattened arrays

  // The output indices tensor has shape (3, B*6F):
  //   - First row: batch indices
  //   - Second row: row indices (within each operator, range 0 to 2F-1)
  //   - Third row: column indices (vertex indices)
  // Fill in batch indices (same for all 6 entries for this face)
  indices[0 * (B * F * 6) + base + 0] = b;
  indices[0 * (B * F * 6) + base + 1] = b;
  indices[0 * (B * F * 6) + base + 2] = b;
  indices[0 * (B * F * 6) + base + 3] = b;
  indices[0 * (B * F * 6) + base + 4] = b;
  indices[0 * (B * F * 6) + base + 5] = b;

  // Fill in row indices (each face contributes two rows)
  indices[1 * (B * F * 6) + base + 0] = 2 * f;      // row for g0_x (vertex v0)
  indices[1 * (B * F * 6) + base + 1] = 2 * f;      // row for g1_x (vertex v1)
  indices[1 * (B * F * 6) + base + 2] = 2 * f;      // row for g2_x (vertex v2)
  indices[1 * (B * F * 6) + base + 3] = 2 * f + 1;  // row for g0_y (vertex v0)
  indices[1 * (B * F * 6) + base + 4] = 2 * f + 1;  // row for g1_y (vertex v1)
  indices[1 * (B * F * 6) + base + 5] = 2 * f + 1;  // row for g2_y (vertex v2)

  // Fill in column indices (vertex indices come from the face connectivity)
  indices[2 * (B * F * 6) + base + 0] = v0;
  indices[2 * (B * F * 6) + base + 1] = v1;
  indices[2 * (B * F * 6) + base + 2] = v2;
  indices[2 * (B * F * 6) + base + 3] = v0;
  indices[2 * (B * F * 6) + base + 4] = v1;
  indices[2 * (B * F * 6) + base + 5] = v2;

  values[base + 0] = g0_x;
  values[base + 1] = g1_x;
  values[base + 2] = g2_x;
  values[base + 3] = g0_y;
  values[base + 4] = g1_y;
  values[base + 5] = g2_y;
}

void intrinsic_gradient_batched_cuda(
    torch::Tensor edge_lengths,  // (B, F, 3)
    torch::Tensor faces,         // (B, F, 3)
    torch::Tensor indices,       // (3, B*6F)
    torch::Tensor values         // (B*6F)
) {
  const int64_t B = edge_lengths.size(0);
  const int64_t F = edge_lengths.size(1);
  const int64_t total_faces = B * F;
  const int threads = 256;
  const int blocks = static_cast<int>((total_faces + threads - 1) / threads);

  c10::cuda::CUDAGuard guard(edge_lengths.device());
  auto stream = at::cuda::getCurrentCUDAStream(edge_lengths.device().index());

  AT_DISPATCH_FLOATING_TYPES(edge_lengths.scalar_type(), "intrinsic_gradient_batched_cuda", [&] {
    using scalar_t_ = scalar_t;
    if (faces.scalar_type() == at::kLong) {
      intrinsic_gradient_batched_kernel<scalar_t_, int64_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          edge_lengths.data_ptr<scalar_t_>(),
          faces.data_ptr<int64_t>(),
          indices.data_ptr<int64_t>(),
          values.data_ptr<scalar_t_>(),
          B,
          F);
    } else if (faces.scalar_type() == at::kInt) {
      intrinsic_gradient_batched_kernel<scalar_t_, int32_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          edge_lengths.data_ptr<scalar_t_>(),
          faces.data_ptr<int32_t>(),
          indices.data_ptr<int64_t>(),
          values.data_ptr<scalar_t_>(),
          B,
          F);
    } else {
      TORCH_CHECK(false, "faces must be int32 or int64");
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// for each batch b and face f, we compute the 2×3 gradient block and
// store six entries (two rows, three columns) in the output sparse operator.
torch::Tensor intrinsic_gradient_operator_batched(
    torch::Tensor edge_lengths, // (B, F, 3) float32/float64
    torch::Tensor faces         // (B, F, 3) int64/int32
) {
  TORCH_CHECK(edge_lengths.is_cuda(), "edge_lengths must be a CUDA tensor");
  TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
  TORCH_CHECK(edge_lengths.dim() == 3 && edge_lengths.size(2) == 3,
              "edge_lengths must be of shape (B, F, 3)");
  TORCH_CHECK(faces.dim() == 3 && faces.size(2) == 3,
              "faces must be of shape (B, F, 3)");

  edge_lengths = edge_lengths.contiguous();
  faces = faces.contiguous();

  const int64_t B = edge_lengths.size(0);
  const int64_t F = edge_lengths.size(1);
  // compute V from the maximum vertex index over all batches (assumes all batches use the same V)
  const int64_t V = faces.max().item<int64_t>() + 1;

  // for a batched sparse tensor with shape (B, 2F, V), the indices tensor has shape (3, B*6F)
  auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(edge_lengths.device());
  torch::Tensor indices = torch::empty({3, B * F * 6}, options_int);

  // allocate the nonzero values (6 per face)
  auto options_float = torch::TensorOptions().dtype(edge_lengths.dtype()).device(edge_lengths.device());
  torch::Tensor values = torch::empty({B * F * 6}, options_float);

  intrinsic_gradient_batched_cuda(edge_lengths, faces, indices, values);
  std::vector<int64_t> size = {B, F * 2, V};
  auto grad_operator = torch::sparse_coo_tensor(indices, values, size);
  return grad_operator;
}

void init_intrinsic_gradient_batched(py::module &m) {
    m.def("intrinsic_gradient_batched",
          &intrinsic_gradient_operator_batched,
          "Intrinsic gradient operator (CUDA)");
  }
  
