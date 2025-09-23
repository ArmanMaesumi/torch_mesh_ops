// bindings.cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

void init_cotangent_laplacian(py::module &m);
void init_cotangent_laplacian_batched(py::module &m);

void init_intrinsic_gradient(py::module &m);
void init_intrinsic_gradient_batched(py::module &m);

void init_intrinsic_gradient_stacked(py::module &m);
void init_intrinsic_gradient_stacked_batched(py::module &m);

void init_vertex_mass(py::module &m);
void init_vertex_mass_batched(py::module &m);

void init_face_areas(py::module &m);
void init_face_areas_batched(py::module &m);

void init_edge_lengths(py::module &m);
void init_edge_lengths_batched(py::module &m);

PYBIND11_MODULE(_torch_mesh_ops, m) {
    init_cotangent_laplacian(m);
    init_cotangent_laplacian_batched(m);
    
    init_intrinsic_gradient(m);
    init_intrinsic_gradient_batched(m);

    init_intrinsic_gradient_stacked(m);
    init_intrinsic_gradient_stacked_batched(m);

    init_vertex_mass(m);
    init_vertex_mass_batched(m);
    
    init_face_areas(m);
    init_face_areas_batched(m);
    
    init_edge_lengths(m);
    init_edge_lengths_batched(m);
}
