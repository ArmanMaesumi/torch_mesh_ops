# Torch Mesh Operators (TMO): A PyTorch CUDA extension for discrete differential operators on triangle meshes.

<!-- ![](./assets/torch_mesh_ops_logo.png) -->
<!-- <img src="./assets/torch_mesh_ops_logo.png" width="450" align="left" style="margin-right:20px"/> -->
<p align="center">
  <img src="./assets/torch_mesh_ops_logo.png" width="350"/>
</p>
This repository offers a PyTorch CUDA extension for constructing discrete differential operators on triangle meshes *directly* on the GPU, significantly speeding up their construction compared to CPU implementations.

These kernels were developed to accelerate our work in [PoissonNet: A Local-Global Approach for Learning on Surfaces](https://github.com/ArmanMaesumi/poissonnet). They are especially useful when mesh operators are needed, e.g., inside of a training loop. Fast GPU-based operator construction eliminates the need to cache operators before training. 

TMO directly emits sparse operators using PyTorch's Sparse COO representation, and supports batching for homogeneous batches of meshes.

## Installation
With PyTorch already installed, you can compile the extension with:
```bash
git clone https://github.com/ArmanMaesumi/torch_mesh_ops
cd torch_mesh_ops
python setup.py install
```

## Notes

**Batched operations for all of the below functions are available through `<function>_batched(...)`, these functions expect an additional batch dimension at the beginning of the input tensors, e.g. `verts: (nB, nV, 3)`, `faces: (nB, nF, 3)`.**

**Currently this package does not support batching for meshes with different numbers of vertices or faces.**

⚠️  ***Warning: To compile `torch_mesh_ops`, the version of your locally installed CUDA Toolkit (the nvcc compiler) must match the CUDA version PyTorch was built with (see torch.version.cuda)***

## Usage
Input tensors below are assumed to live on the GPU. Floating-point tensors can be either `torch.float` or `torch.double`, and index tensors can be either `torch.int` or `torch.long`. It is generally recommended to use `fp64` precision to avoid numerical degeneracies.

```python
import torch
import torch_mesh_ops as TMO
...
```

#### Cotangent Laplacian
```python
L = TMO.cotangent_laplacian(verts, faces, denom_eps=1e-8)
```
Returns a sparse matrix of shape (nV, nV) representing the Laplacian matrix (no area weighting).

- verts (torch.Tensor): (nV, 3) Float tensor of vertex positions
- faces (torch.Tensor): (nF, 3) Integer tensor of triangle indices
- denom_eps (optional float): stabilization term for degenerate faces (default 0.0)

#### Intrinsic Gradient
```python
grad = TMO.intrinsic_gradient(edge_lens, faces)
```
Returns a sparse matrix of shape (2*nF, nV) representing the intrinsic gradient operator defined by edge lengths. This operator maps vertex-based scalar fields to their face-based tangent gradients. The elements of the first dimension ***interleave*** the x and y components of the gradient. For convenience, we also provide `TMO.intrinsic_gradient_stacked` If you prefer the x and y components to be *stacked*.

- edge_lens (torch.Tensor): (nF, 3) Float tensor of edge lengths (see `TMO.edge_lengths` for convention)
- faces (torch.Tensor): (nF, 3) Integer tensor of triangle indices

#### Vertex Mass
```python
mass = TMO.vertex_mass(verts, faces, eps=1e-8)
```
Returns a vector of lumped vertex masses with shape `(nV,)` as accumulation of one third of adjacent face areas.

- verts (torch.Tensor): (nV, 3) Float tensor of vertex positions
- faces (torch.Tensor): (nF, 3) Integer tensor of triangle indices
- eps (optional float): regularization term that updates result via `mass += eps * mean(mass)` (default 0.0)

#### Face Areas
```python
areas = TMO.face_areas(verts, faces)
```
Returns vector of face areas with shape `(nF,)`.

- verts (torch.Tensor): (nV, 3) Float tensor of vertex positions
- faces (torch.Tensor): (nF, 3) Integer tensor of triangle indices

#### Edge Lengths
```python
lengths = TMO.edge_lengths(verts, faces)
```
Returns matrix of edge lengths with shape `(nF, 3)`, the last dimension holds entries for edges `[1,2], [2,0], [0, 1]` on each face (i.e. edges opposite to vertices i,j,k).

- verts (torch.Tensor): (nV, 3) Float tensor of vertex positions
- faces (torch.Tensor): (nF, 3) Integer tensor of triangle indices

## Troubleshooting:
Runtime error:
```
ImportError: libc10.so: cannot open shared object file: No such file or directory
```
Solution: Update `LD_LIBRARY_PATH` to tell dynamic linker where to find libc10.so
```bash
export LD_LIBRARY_PATH=$(python -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

<br/>

Installation error:
```
The detected CUDA version ... mismatches the version that was used to compile PyTorch ... Please make sure to use the same CUDA versions
```
Solution: Your `nvcc` version does not match the CUDA runtime version used by PyTorch. For example, if you installed CUDA Toolkit 12.8, then you need PyTorch+cu128 to install our package. You can use `which nvcc && nvcc --version` and compare the output to `import torch; torch.version.cuda`.

<br/>

Installation error on Windows:
```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"
```
Solution: Download [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). In the installer select "Desktop development with C++" and click install. For more help see [here](https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst).

## Citation
This package was developed during our work on [PoissonNet](https://github.com/poissonnet). If you found this package helpful, please considering citing as:
```bibtex
@article{maesumi2025poissonnet,
author = {Maesumi, Arman and Makadia, Tanish and Groueix, Thibault and Kim, Vladimir G. and Ritchie, Daniel and Aigerman, Noam},
title = {PoissonNet: A Local-Global Approach for Learning on Surfaces},
year = {2025},
booktitle = {ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia 2025)},
publisher = {Association for Computing Machinery}
}
```

## Acknowledgements

The development of these CUDA kernels greatly benefited from several public resources. In particular, [potpourri3d](https://github.com/nmwsharp/potpourri3d) and [libigl](https://github.com/libigl/libigl) were invaluable references for the construction of operators.

