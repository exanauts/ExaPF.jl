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
KrylovBICGSTAB
DQGMRES
BICGSTAB
EigenBICGSTAB
```

`ExaPF.jl` is shipped with a custom BICGSTAB implementation.
However, we highly recommend to use `KrylovBICGSTAB` instead,
which has proved to be more robust.
```@docs
bicgstab

```

Available linear solvers can be queried with
```@docs
list_solvers

```
A default solver is provided for each vendor backend.
```@docs
default_linear_solver

```

## Preconditioning

To solve linear systems with iterative methods, `ExaPF`
provides an implementation of a block-Jacobi preconditioner,
portable on GPU.

```@docs
AbstractPreconditioner
```

### Block-Jacobi preconditioner

```@docs
BlockJacobiPreconditioner
update
```
