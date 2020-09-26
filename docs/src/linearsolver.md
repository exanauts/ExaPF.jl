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

As mentioned before, a linear solver is required to compute the Newton step in 

```julia
dx .= jacobian(x)\f(x)
```

Our package supports the following linear solvers:

* CUSOLVER with `csrlsvqr` (GPU),
* `Krylov.jl` with `dqgmres` (CPU/GPU), 
* `IterativeSolvers` with `bicgstab` (CPU) [@sleijpen1993bicgstab],
* UMFPACK through the default Julia `\` operator (CPU),
* and a custom BiCGSTAB implementation [@bicgstabVorst] \(CPU/GPU\).

The last custom implementation was necessary as BiCGSTAB showed much better
performance than GMRES and at the time of this writing both `Krylov.jl` and
`IterativeSolvers.jl` did not provide an implementation that supported
`CUDA.jl`.

Using only an iterative solver lead to divergence and bad performance due to
ill-conditioning of the Jacobian. This is a known phenomenon in power
systems. That's why this package comes with a block Jacobi preconditioner
that is tailored towards GPUs and is proven to work well with power flow
problems.

The Jacobian is partitioned into a dense block diagonal structure, where each block is inverted to build our preconditioner `P`. For the partition we use `Metis.jl`.

![Dense block Jacobi preconditioner \label{fig:preconditioner}](figures/gpublocks.png)

Compared to incomplete Cholesky and incomplete LU this preconditioner is easily portable to the GPU if the number of blocks is high enough. `ExaPF.jl` uses the batch BLAS calls from `CUBLAS` to invert the single blocks.

```julia
CUDA.@sync pivot, info = CUDA.CUBLAS.getrf_batched!(blocks, true)
CUDA.@sync pivot, info, p.cuJs = CUDA.CUBLAS.getri_batched(blocks, pivot)
```

Assuming that other vendors will provide such batched BLAS APIs, this code is portable to other GPU architectures.

## Description
```@docs
Precondition.AbstractPreconditioner
```

## API Reference
```@docs
Precondition.Preconditioner
Precondition.update
Precondition.build_adjmatrix
Precondition.fillblock_gpu!
Precondition.fillP_gpu!
```