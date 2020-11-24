```@meta
CurrentModule = ExaPF.LinearSolvers
```

## Linear solvers

`ExaPF` allows to solve linear systems with either
direct and indirect linear algebra.
```@docs
ldiv!

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
