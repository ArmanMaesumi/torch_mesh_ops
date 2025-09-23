#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// Each thread processes one face (per batch), computing all three corner cotangents
template <typename scalar_t, typename index_t>
__global__ void cotan_laplacian_faces_kernel_batched(
    const scalar_t* __restrict__ V, // shape: [B, nV, 3]
    const index_t*  __restrict__ F, // shape: [B, nF, 3]
    int64_t B,                      // number of batches
    int64_t nF,                     // number of faces per mesh
    int64_t nV,                     // number of vertices per mesh
    scalar_t eps2,
    // out_indices: shape [3, 12 * B * nF] flattened; values: [12 * B * nF]
    int64_t* __restrict__ out_indices,
    scalar_t* __restrict__ out_values) {

  const int64_t face_global = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_faces = B * nF;
  if (face_global >= total_faces) return;

  const int64_t b = face_global / nF;   // batch index
  const int64_t f = face_global % nF;   // face index within batch

  const scalar_t* Vb = V + b * nV * 3;
  const index_t* Fb = F + b * nF * 3;

  // vertex indices for this face
  const index_t i0 = Fb[f * 3 + 0];
  const index_t i1 = Fb[f * 3 + 1];
  const index_t i2 = Fb[f * 3 + 2];

  // vertex positions
  const scalar_t p0x = Vb[i0 * 3 + 0];
  const scalar_t p0y = Vb[i0 * 3 + 1];
  const scalar_t p0z = Vb[i0 * 3 + 2];
  const scalar_t p1x = Vb[i1 * 3 + 0];
  const scalar_t p1y = Vb[i1 * 3 + 1];
  const scalar_t p1z = Vb[i1 * 3 + 2];
  const scalar_t p2x = Vb[i2 * 3 + 0];
  const scalar_t p2y = Vb[i2 * 3 + 1];
  const scalar_t p2z = Vb[i2 * 3 + 2];

  // computes 0.5*cot(angle at A) using vectors AB and AC
  auto half_cot = [&](scalar_t abx, scalar_t aby, scalar_t abz,
                      scalar_t acx, scalar_t acy, scalar_t acz) -> scalar_t {
    const scalar_t dot = abx * acx + aby * acy + abz * acz;
    const scalar_t cx = aby * acz - abz * acy;
    const scalar_t cy = abz * acx - abx * acz;
    const scalar_t cz = abx * acy - aby * acx;
    const scalar_t cross2 = cx * cx + cy * cy + cz * cz;
    const scalar_t inv_cross = rsqrt(cross2 + eps2);
    return static_cast<scalar_t>(0.5) * dot * inv_cross;
  };

  // Compute 0.5*cotangents at each vertex of the triangle
  // angle at v0 uses vectors (v1 - v0) and (v2 - v0)
  const scalar_t c0 = half_cot(p1x - p0x, p1y - p0y, p1z - p0z,
                               p2x - p0x, p2y - p0y, p2z - p0z);
  // angle at v1 uses vectors (v2 - v1) and (v0 - v1)
  const scalar_t c1 = half_cot(p2x - p1x, p2y - p1y, p2z - p1z,
                               p0x - p1x, p0y - p1y, p0z - p1z);
  // angle at v2 uses vectors (v0 - v2) and (v1 - v2)
  const scalar_t c2 = half_cot(p0x - p2x, p0y - p2y, p0z - p2z,
                               p1x - p2x, p1y - p2y, p1z - p2z);

  // output pointers
  const int64_t total_entries = B * nF * 12;
  const int64_t base = b * nF * 12 + f * 12;              // 12 entries per face
  int64_t* batch_ptr = out_indices;                      // indices[0, :]
  int64_t* row_ptr   = out_indices + total_entries;      // indices[1, :]
  int64_t* col_ptr   = out_indices + 2 * total_entries;  // indices[2, :]

  // writes 4 contributions for the angle opposite edge (u,v)
  auto emit = [&](int64_t& off, index_t u, index_t v, scalar_t w) {
    batch_ptr[base + off + 0] = b; row_ptr[base + off + 0] = u; col_ptr[base + off + 0] = u; out_values[base + off + 0] =  w;
    batch_ptr[base + off + 1] = b; row_ptr[base + off + 1] = v; col_ptr[base + off + 1] = v; out_values[base + off + 1] =  w;
    batch_ptr[base + off + 2] = b; row_ptr[base + off + 2] = u; col_ptr[base + off + 2] = v; out_values[base + off + 2] = -w;
    batch_ptr[base + off + 3] = b; row_ptr[base + off + 3] = v; col_ptr[base + off + 3] = u; out_values[base + off + 3] = -w;
    off += 4;
  };

  int64_t off = 0;
  // Angle at v0 contributes to pair (v1, v2)
  emit(off, i1, i2, c0);
  // Angle at v1 contributes to pair (v2, v0)
  emit(off, i2, i0, c1);
  // Angle at v2 contributes to pair (v0, v1)
  emit(off, i0, i1, c2);
}

torch::Tensor cotan_laplacian_cuda_batched(torch::Tensor V, torch::Tensor F, double denom_eps, bool do_coalesce) {
  TORCH_CHECK(V.is_cuda() && F.is_cuda(), "V and F must be CUDA tensors");
  TORCH_CHECK(V.dim()==3 && V.size(2)==3, "V must be [B,nV,3]");
  TORCH_CHECK(F.dim()==3 && F.size(2)==3, "F must be [B,nF,3]");

  V = V.contiguous();
  F = F.contiguous();

  // V: [B, nV, 3] and F: [B, nF, 3]
  const int64_t B  = V.size(0);
  const int64_t nV = V.size(1);
  const int64_t nF = F.size(1);

  // each face produces 12 entries
  const int64_t num_entries = B * nF * 12;
  auto indices = torch::empty({3, num_entries}, V.options().dtype(torch::kInt64));
  auto values  = torch::empty({num_entries}, V.options());

  c10::cuda::CUDAGuard guard(V.device());
  auto stream = at::cuda::getCurrentCUDAStream(V.device().index());

  const int total_threads = static_cast<int>(B * nF);
  const int threads = 256;
  const int blocks = (total_threads + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(V.scalar_type(), "cotan_laplacian_batched", [&]{
    using scalar_t_ = scalar_t;
    const scalar_t_ eps2 = static_cast<scalar_t_>(denom_eps * denom_eps);

    if (F.scalar_type() == at::kLong) {
      cotan_laplacian_faces_kernel_batched<scalar_t_, int64_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          V.data_ptr<scalar_t_>(), F.data_ptr<int64_t>(), B, nF, nV, eps2,
          indices.data_ptr<int64_t>(), values.data_ptr<scalar_t_>());
    } else if (F.scalar_type() == at::kInt) {
      cotan_laplacian_faces_kernel_batched<scalar_t_, int32_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          V.data_ptr<scalar_t_>(), F.data_ptr<int32_t>(), B, nF, nV, eps2,
          indices.data_ptr<int64_t>(), values.data_ptr<scalar_t_>());
    } else {
      TORCH_CHECK(false, "F must be int32 or int64");
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto L = torch::sparse_coo_tensor(indices, values, {B, nV, nV});
  return do_coalesce ? L.coalesce() : L;
}

torch::Tensor cotan_laplacian_batched(torch::Tensor V, torch::Tensor F, double denom_eps = 0.0, bool coalesce = true) {
  if (V.device().is_cuda()) {
    return cotan_laplacian_cuda_batched(V, F, denom_eps, coalesce);
  }
  AT_ERROR("CPU version not implemented");
}

void init_cotangent_laplacian_batched(py::module &m) {
  m.def("cotangent_laplacian_batched", &cotan_laplacian_batched, "Batched Cotangent Laplacian operator (CUDA)",
        py::arg("V"), py::arg("F"), py::arg("denom_eps") = 0.0, py::arg("coalesce") = true);
}
