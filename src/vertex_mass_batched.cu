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
__global__ void compute_vertex_areas_kernel_batched(
    const scalar_t* __restrict__ V, // shape: (B, nV, 3)
    int64_t nV,
    const index_t* __restrict__ F,  // shape: (B, nF, 3)
    int64_t nF,
    int64_t B,
    scalar_t* __restrict__ vertex_areas // shape: (B, nV)
) {
    // global face index across all batches
    int64_t global_face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_faces = B * nF;
    if (global_face_idx >= total_faces)
        return;

    int64_t b = global_face_idx / nF;           // batch index
    int64_t face_idx = global_face_idx % nF;    // face index within batch b

    // Fetch the vertex indices for this face
    // F is stored contiguously with layout: [B, nF, 3]
    index_t i0 = F[b * nF * 3 + face_idx * 3 + 0];
    index_t i1 = F[b * nF * 3 + face_idx * 3 + 1];
    index_t i2 = F[b * nF * 3 + face_idx * 3 + 2];

    // vertex positions
    scalar_t v0x = V[b * nV * 3 + static_cast<int64_t>(i0) * 3 + 0];
    scalar_t v0y = V[b * nV * 3 + static_cast<int64_t>(i0) * 3 + 1];
    scalar_t v0z = V[b * nV * 3 + static_cast<int64_t>(i0) * 3 + 2];
    scalar_t v1x = V[b * nV * 3 + static_cast<int64_t>(i1) * 3 + 0];
    scalar_t v1y = V[b * nV * 3 + static_cast<int64_t>(i1) * 3 + 1];
    scalar_t v1z = V[b * nV * 3 + static_cast<int64_t>(i1) * 3 + 2];
    scalar_t v2x = V[b * nV * 3 + static_cast<int64_t>(i2) * 3 + 0];
    scalar_t v2y = V[b * nV * 3 + static_cast<int64_t>(i2) * 3 + 1];
    scalar_t v2z = V[b * nV * 3 + static_cast<int64_t>(i2) * 3 + 2];

    // edge vectors from v0 to v1 and v0 to v2
    scalar_t e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    scalar_t e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

    // cross product of e1 and e2
    scalar_t cx = e1y * e2z - e1z * e2y;
    scalar_t cy = e1z * e2x - e1x * e2z;
    scalar_t cz = e1x * e2y - e1y * e2x;

    scalar_t area = static_cast<scalar_t>(0.5) * sqrt(cx * cx + cy * cy + cz * cz);
    scalar_t area_third = area / static_cast<scalar_t>(3.0);

    // accumulate one-third of the face area to each of the three vertices
    // vertex i in batch b is at index (b * nV + i)
    atomicAdd(&vertex_areas[b * nV + static_cast<int64_t>(i0)], area_third);
    atomicAdd(&vertex_areas[b * nV + static_cast<int64_t>(i1)], area_third);
    atomicAdd(&vertex_areas[b * nV + static_cast<int64_t>(i2)], area_third);
}

torch::Tensor vertex_areas_cuda_batched(torch::Tensor V, torch::Tensor F, double eps = 0.0) {
    TORCH_CHECK(V.is_cuda() && F.is_cuda(), "V and F must be CUDA tensors");
    TORCH_CHECK(V.dim()==3 && V.size(2)==3, "V must be [B,nV,3]");
    TORCH_CHECK(F.dim()==3 && F.size(2)==3, "F must be [B,nF,3]");

    V = V.contiguous();
    F = F.contiguous();

    const int64_t B  = V.size(0);
    const int64_t nV = V.size(1);
    const int64_t nF = F.size(1);

    auto vertex_areas = torch::zeros({B, nV}, V.options());

    // launch one thread per face across all batches
    const int64_t total_faces = B * nF;
    const int threads = 256;
    const int blocks = static_cast<int>((total_faces + threads - 1) / threads);

    c10::cuda::CUDAGuard guard(V.device());
    auto stream = at::cuda::getCurrentCUDAStream(V.device().index());

    AT_DISPATCH_FLOATING_TYPES(V.scalar_type(), "vertex_areas_cuda_batched", [&] {
        using scalar_t_ = scalar_t;
        if (F.scalar_type() == at::kLong) {
            compute_vertex_areas_kernel_batched<scalar_t_, int64_t><<<blocks, threads, 0, stream.stream()>>>(
                V.data_ptr<scalar_t_>(),
                nV,
                F.data_ptr<int64_t>(),
                nF,
                B,
                vertex_areas.data_ptr<scalar_t_>()
            );
        } else if (F.scalar_type() == at::kInt) {
            compute_vertex_areas_kernel_batched<scalar_t_, int32_t><<<blocks, threads, 0, stream.stream()>>>(
                V.data_ptr<scalar_t_>(),
                nV,
                F.data_ptr<int32_t>(),
                nF,
                B,
                vertex_areas.data_ptr<scalar_t_>()
            );
        } else {
            TORCH_CHECK(false, "F must be int32 or int64");
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Optionally add a small eps–weighted per-mesh mean area to each vertex
    if (eps != 0.0) {
        auto mean_area = vertex_areas.mean(1, true);
        vertex_areas.add_(eps * mean_area);
    }
    return vertex_areas;
}

void init_vertex_mass_batched(py::module &m) {
  m.def("vertex_mass_batched", &vertex_areas_cuda_batched,
        "Compute lumped vertex masses (areas) for batched meshes using CUDA with optional eps addition\n"
        "V: (B, nV, 3) vertex positions\n"
        "F: (B, nF, 3) face indices\n"
        "eps: optional scalar",
        py::arg("V"), py::arg("F"), py::arg("eps") = 0.0);
}
