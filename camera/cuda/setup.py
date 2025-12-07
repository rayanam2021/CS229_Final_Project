from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ray_tracing_cuda',
    ext_modules=[
        CUDAExtension(
            name='ray_tracing_cuda',
            sources=['ray_tracing_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    '--ptxas-options=-v',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
