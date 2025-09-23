# setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torch_mesh_ops',
    version='1.0.0',
    description='PyTorch CUDA extension for construction of discrete differential operators on triangle meshes.',
    author='Arman Maesumi',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='torch_mesh_ops._torch_mesh_ops',
            sources=[
                'src/bindings.cpp',

                'src/cotangent_laplacian.cu',
                'src/cotangent_laplacian_batched.cu',

                'src/vertex_mass.cu',
                'src/vertex_mass_batched.cu',

                'src/intrinsic_gradient.cu',
                'src/intrinsic_gradient_batched.cu',

                'src/intrinsic_gradient_stacked.cu',
                'src/intrinsic_gradient_stacked_batched.cu',

                'src/face_areas.cu',
                'src/face_areas_batched.cu',

                'src/edge_lengths.cu',
                'src/edge_lengths_batched.cu'
            ],
            extra_compile_args={'cxx': ['-O3'],
                                'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch'],
    zip_safe=False
)
