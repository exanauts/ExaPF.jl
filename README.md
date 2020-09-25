# ExaPF

| **Test Status** |
|:----------------:|
![Run tests](https://github.com/exanauts/ExaPF.jl/workflows/Run%20tests/badge.svg?branch=dev%2Frgm) |

ExaPF is a HPC package for solving power flow (PF) on a GPU. It currently solves PF using the Newton-Raphson algorithm on NVIDIA GPUs.
Its main features are:

* Using [CUDA.jl](https://juliagpu.gitlab.io/CUDA.jl/) CuArrays arrays for generating CUDA kernels using the broadcast '.' operator.
* Using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and Jacobian coloring to generate the compressed Jacobian of the PF equations. The Jacobian evaluation is taking place fully on the GPU.
* Preconditioned BICGSTAB implemented in CuArrays as the iterative solver.
* A block Jacobi preconditioner that updates on the GPU.

This code will serve as the basis for OPF on GPUs using the reduced gradient method. A similar abstraction than CuArrays will be used to port the code to AMD ROCm and Intel oneAPI.

## Installation

The package is not yet available through the Julia registrator.

```julia
pkg> dev git@github.com:exanauts/ExaPF.jl.git
```

## Test
```julia
pkg> test ExaPF
```

## Documentation
WIP


[build-img]: https://travis-ci.com/exanauts/ExaPF.jl.svg?branch=master
[build-url]: https://travis-ci.com/exanauts/ExaPF.jl

