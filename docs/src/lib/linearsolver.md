```@meta
CurrentModule = ExaPF.LinearSolvers
```

## Linear solvers

`ExaPF` allows to solve linear systems with either
direct and indirect linear algebra.
```@docs
ldiv!
DirectSolver
KrylovBICGSTAB
BICGSTAB
EigenBICGSTAB

```

Available linear solvers could be queried with
```@docs
list_solvers

```

`ExaPF.jl` is shipped with a custom BICGSTAB implementation.
However, we highly recommend to use `KrylovBICGSTAB` instead,
which has proved to be more robust.
```@docs
bicgstab

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
build_adjmatrix
fillblock_gpu!
fillP_gpu!
```
