#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// each thread computes contribution of one face to adjacent vertices
template <typename scalar_t, typename index_t>
__global__ void compute_vertex_areas_kernel(
    const scalar_t* __restrict__ V,
    int64_t nV,
    const index_t* __restrict__ F,
    int64_t nF,
    scalar_t* __restrict__ vertex_areas
) {
    int64_t face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= nF) return;

    index_t i0 = F[face_idx * 3 + 0];
    index_t i1 = F[face_idx * 3 + 1];
    index_t i2 = F[face_idx * 3 + 2];

    scalar_t v0x = V[static_cast<int64_t>(i0) * 3 + 0];
    scalar_t v0y = V[static_cast<int64_t>(i0) * 3 + 1];
    scalar_t v0z = V[static_cast<int64_t>(i0) * 3 + 2];
    scalar_t v1x = V[static_cast<int64_t>(i1) * 3 + 0];
    scalar_t v1y = V[static_cast<int64_t>(i1) * 3 + 1];
    scalar_t v1z = V[static_cast<int64_t>(i1) * 3 + 2];
    scalar_t v2x = V[static_cast<int64_t>(i2) * 3 + 0];
    scalar_t v2y = V[static_cast<int64_t>(i2) * 3 + 1];
    scalar_t v2z = V[static_cast<int64_t>(i2) * 3 + 2];

    scalar_t e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    scalar_t e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

    scalar_t cx = e1y * e2z - e1z * e2y;
    scalar_t cy = e1z * e2x - e1x * e2z;
    scalar_t cz = e1x * e2y - e1y * e2x;

    scalar_t area = static_cast<scalar_t>(0.5) * sqrt(cx * cx + cy * cy + cz * cz);
    scalar_t area_third = area / static_cast<scalar_t>(3.0);

    atomicAdd(&vertex_areas[static_cast<int64_t>(i0)], area_third);
    atomicAdd(&vertex_areas[static_cast<int64_t>(i1)], area_third);
    atomicAdd(&vertex_areas[static_cast<int64_t>(i2)], area_third);
}

torch::Tensor vertex_areas_cuda(torch::Tensor V, torch::Tensor F, double eps = 0.0) {
    TORCH_CHECK(V.is_cuda() && F.is_cuda(), "V and F must be CUDA tensors");
    TORCH_CHECK(V.dim()==2 && V.size(1)==3, "V must be [nV,3]");
    TORCH_CHECK(F.dim()==2 && F.size(1)==3, "F must be [nF,3]");

    V = V.contiguous();
    F = F.contiguous();

    const int64_t nV = V.size(0);
    const int64_t nF = F.size(0);

    auto vertex_areas = torch::zeros({nV}, V.options());

    const int threads = 256;
    const int blocks = static_cast<int>((nF + threads - 1) / threads);

    c10::cuda::CUDAGuard guard(V.device());
    auto stream = at::cuda::getCurrentCUDAStream(V.device().index());

    AT_DISPATCH_FLOATING_TYPES(V.scalar_type(), "vertex_areas_cuda", [&] {
        using scalar_t_ = scalar_t;
        if (F.scalar_type() == at::kLong) {
            compute_vertex_areas_kernel<scalar_t_, int64_t><<<blocks, threads, 0, stream.stream()>>>(
                V.data_ptr<scalar_t_>(),
                nV,
                F.data_ptr<int64_t>(),
                nF,
                vertex_areas.data_ptr<scalar_t_>()
            );
        } else if (F.scalar_type() == at::kInt) {
            compute_vertex_areas_kernel<scalar_t_, int32_t><<<blocks, threads, 0, stream.stream()>>>(
                V.data_ptr<scalar_t_>(),
                nV,
                F.data_ptr<int32_t>(),
                nF,
                vertex_areas.data_ptr<scalar_t_>()
            );
        } else {
            TORCH_CHECK(false, "F must be int32 or int64");
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (eps != 0.0) {
        auto mean_area = vertex_areas.mean();
        vertex_areas.add_(eps * mean_area);
    }
    return vertex_areas;
}

void init_vertex_mass(py::module &m) {
  m.def("vertex_mass", &vertex_areas_cuda,
        "Compute lumped vertex masses (areas) using CUDA with optional eps addition",
        py::arg("V"), py::arg("F"), py::arg("eps") = 0.0);
}
