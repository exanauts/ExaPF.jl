```@meta
CurrentModule = ExaPF.LinearSolvers
```

# Linear solvers

## Description
`ExaPF` allows to solve linear systems with either
direct and indirect linear algebra, both on CPU and on GPU.
To solve a linear system $Ax = b$, `ExaPF` uses the function `ldiv!`.
```@docs
ldiv!
```

## Direct solvers

`ExaPF` wraps UMFPACK (shipped with `SuiteSparse.jl`) on the CPU,
and CUSPARSE on CUDA device.

```@docs
DirectSolver
```

## Iterative solvers

```@docs
Bicgstab
Dqgmres
```

Available linear solvers can be queried with
```@docs
list_solvers

```
A default solver is provided for each vendor backend.
```@docs
default_linear_solver

```

## Block-Krylov solvers

`ExaPF.jl` provides an implementation of block-GMRES to solve linear systems
with multiple righ-hand sides.

```@docs
BlockKrylovSolver
BlockGmresSolver
block_gmres
block_gmres!
```
