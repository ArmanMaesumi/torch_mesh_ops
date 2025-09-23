#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

template <typename scalar_t, typename index_t>
__global__ void cotan_laplacian_per_face_kernel(
    const scalar_t* __restrict__ V,   // [nV,3]
    const index_t*  __restrict__ F,   // [nF,3]
    int64_t nF,
    scalar_t eps2,
    int64_t* __restrict__ out_idx,    // [2, 12*nF]
    scalar_t* __restrict__ out_val) { // [12*nF]
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= nF) return;

  index_t i = F[3*f+0], j = F[3*f+1], k = F[3*f+2];

  // Load positions
  scalar_t ix = V[3*i+0], iy = V[3*i+1], iz = V[3*i+2];
  scalar_t jx = V[3*j+0], jy = V[3*j+1], jz = V[3*j+2];
  scalar_t kx = V[3*k+0], ky = V[3*k+1], kz = V[3*k+2];

  auto cot_half = [&](scalar_t ax, scalar_t ay, scalar_t az,
                      scalar_t bx, scalar_t by, scalar_t bz) {
    scalar_t dot = ax*bx + ay*by + az*bz;
    scalar_t cx = ay*bz - az*by;
    scalar_t cy = az*bx - ax*bz;
    scalar_t cz = ax*by - ay*bx;
    scalar_t cross2 = cx*cx + cy*cy + cz*cz;
    scalar_t inv_cross = rsqrt(cross2 + eps2);
    return static_cast<scalar_t>(0.5) * dot * inv_cross;
  };

  // Opposite to edge (i,j): corner at k -> vectors i-k, j-k
  scalar_t cij = cot_half(ix-kx, iy-ky, iz-kz, jx-kx, jy-ky, jz-kz);
  // Opposite to edge (j,k): corner at i -> vectors j-i, k-i
  scalar_t cjk = cot_half(jx-ix, jy-iy, jz-iz, kx-ix, ky-iy, kz-iz);
  // Opposite to edge (k,i): corner at j -> vectors k-j, i-j
  scalar_t cki = cot_half(kx-jx, ky-jy, kz-jz, ix-jx, iy-jy, iz-jz);

  // Write 12 entries for this face
  int64_t base = f * 12;
  int64_t* rows = out_idx;
  int64_t* cols = out_idx + (nF * 12);

  // For each corner, add { (p,p)+=c, (q,q)+=c, (p,q)-=c, (q,p)-=c }
  auto emit = [&](int64_t& off, index_t p, index_t q, scalar_t c){
    rows[base+off+0] = p; cols[base+off+0] = p; out_val[base+off+0] =  c;
    rows[base+off+1] = q; cols[base+off+1] = q; out_val[base+off+1] =  c;
    rows[base+off+2] = p; cols[base+off+2] = q; out_val[base+off+2] = -c;
    rows[base+off+3] = q; cols[base+off+3] = p; out_val[base+off+3] = -c;
    off += 4;
  };
  int64_t off = 0;
  emit(off, i, j, cij);
  emit(off, j, k, cjk);
  emit(off, k, i, cki);
}


torch::Tensor cotan_laplacian_cuda(torch::Tensor V, torch::Tensor F, double denom_eps) {
  TORCH_CHECK(V.is_cuda() && F.is_cuda(), "V and F must be CUDA tensors");
  TORCH_CHECK(V.dim()==2 && V.size(1)==3, "V must be [nV,3]");
  TORCH_CHECK(F.dim()==2 && F.size(1)==3, "F must be [nF,3]");
  V = V.contiguous();
  F = F.contiguous();

  const auto nV = V.size(0);
  const auto nF = F.size(0);
  const int64_t N = nF * 12;

  auto idx = torch::empty({2, N}, V.options().dtype(torch::kInt64));
  auto val = torch::empty({N}, V.options());

  c10::cuda::CUDAGuard guard(V.device());
  auto stream = at::cuda::getCurrentCUDAStream(V.device().index());

  int threads = 256;
  int blocks  = (nF + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(V.scalar_type(), "cotan_laplacian", [&]{
    using scalar_t_ = scalar_t;
    const scalar_t_ eps2 = static_cast<scalar_t_>(denom_eps * denom_eps);

    if (F.scalar_type() == at::kLong) {
      cotan_laplacian_per_face_kernel<scalar_t_, int64_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          V.data_ptr<scalar_t_>(), F.data_ptr<int64_t>(), nF, eps2,
          idx.data_ptr<int64_t>(), val.data_ptr<scalar_t_>());
    } else if (F.scalar_type() == at::kInt) {
      cotan_laplacian_per_face_kernel<scalar_t_, int32_t>
        <<<blocks, threads, 0, stream.stream()>>>(
          V.data_ptr<scalar_t_>(), F.data_ptr<int32_t>(), nF, eps2,
          idx.data_ptr<int64_t>(), val.data_ptr<scalar_t_>());
    } else {
      TORCH_CHECK(false, "F must be int32 or int64");
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto L = torch::sparse_coo_tensor(idx, val, {nV, nV});
  return L.coalesce();
}

torch::Tensor cotan_laplacian(torch::Tensor V, torch::Tensor F, double denom_eps = 0.) {
  if (V.device().is_cuda()) {
    return cotan_laplacian_cuda(V, F, denom_eps);
  }
  AT_ERROR("CPU version not implemented");
}

void init_cotangent_laplacian(py::module &m) {
  m.def("cotangent_laplacian", &cotan_laplacian, "Cotangent Laplacian operator (CUDA)",
        py::arg("V"), py::arg("F"), py::arg("denom_eps") = 0.0);
}