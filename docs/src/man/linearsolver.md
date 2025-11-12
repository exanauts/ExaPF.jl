```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
    const Precondition = ExaPF.Precondition
    const Iterative = ExaPF.Iterative
end
DocTestFilters = [r"ExaPF"]
```
# Linear Solver

## Overview

As mentioned before, a linear solver is required to compute the Newton step in

```julia
dx .= jacobian(x)\f(x)
```

Our package supports the following linear solvers:

* [UMFPACK](https://github.com/DrTimothyAldenDavis/SuiteSparse) (CPU),
* [KLU](https://github.com/DrTimothyAldenDavis/SuiteSparse) (CPU),
* [`cusolverRF`](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrf-refactorization-reference-deprecated) (NVIDIA GPU),
* [`cuDSS`](https://developer.nvidia.com/cudss) (NVIDIA GPU),
* [`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl) with `dqgmres` and `bicgstab` (CPU, NVIDIA GPU, AMD GPU),
* Any custom linear solver provided by the user.

## Preconditioning

Using only an iterative solver leads to divergence and bad performance due to
ill-conditioning of the Jacobian. This is a known phenomenon in power
systems. That's why this package comes with a block Jacobi preconditioner
that is tailored towards GPUs and is proven to work well with power flow
problems.

The block-Jacobi preconditioner used in ExaPF has been added to [`KrylovPreconditioners.jl`](https://github.com/JuliaSmoothOptimizers/KrylovPreconditioners.jl)

Using Metis.jl, the sparse Jacobian is reordered to expose a dense block-diagonal structure, on which a block-Jacobi preconditioner becomes relevant and efficient.

![METIS \label{fig:metis}](../figures/reordering_jacobian.png)

Subsequently, each diagonal block is treated as dense and inverted to form the block-Jacobi preconditioner `P`.

![Preconditioner \label{fig:preconditioner}](../figures/preconditioner_jacobian.png)

Compared to incomplete Cholesky and incomplete LU this preconditioner is easily portable to the GPU if the number of blocks is high enough. `ExaPF.jl` uses the batch BLAS / LAPACK calls from `cuBLAS / cuSOLVER` or `rocBLAS / rocSOLVER` to invert the single blocks.

```julia
CUDA.@sync pivot, info = CUDA.CUBLAS.getrf_batched!(blocks, true)
CUDA.@sync pivot, info, p.cuJs = CUDA.CUBLAS.getri_batched(blocks, pivot)
```
