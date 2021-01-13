# ExaPF

| **Documentation**                       | **Build Status**                                              |
|:---------------------------------------:|:-------------------------------------------------------------:|
| [![][docs-latest-img]][docs-latest-url] | ![Run tests](https://github.com/exanauts/ExaPF.jl/workflows/Run%20tests/badge.svg?branch=master) | 

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://exanauts.github.io/ExaPF.jl/

ExaPF is a HPC package for solving power flow (PF) on a GPU. It currently solves PF using the Newton-Raphson algorithm on NVIDIA GPUs.
Its main features are:

* Using [CUDA.jl](https://juliagpu.gitlab.io/CUDA.jl/) CuArrays arrays for generating CUDA kernels using the broadcast '.' operator.
* Using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and Jacobian coloring to generate the compressed Jacobian of the PF equations. The Jacobian evaluation is taking place fully on the GPU.
* Preconditioned BICGSTAB with support for [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
* A block Jacobi preconditioner that updates on the GPU.

This code will serve as the basis for OPF on GPUs using the reduced gradient method. A similar abstraction than CuArrays will be used to port the code to AMD ROCm and Intel oneAPI through [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) and [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl), respectively.

## Installation

```julia
pkg> add ExaPF
```

## Test
```julia
pkg> test ExaPF
```

## Funding
This research was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.

