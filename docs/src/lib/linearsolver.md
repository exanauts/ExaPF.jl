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

`ExaPF` wraps KLU on the CPU, and cuDSS on NVIDIA GPU.

```@docs
DirectSolver
```

## Iterative solvers

`ExaPF` wraps BICGSTAB and DQGMRES for CPU, NVIDIA GPU, and AMD GPU.

```@docs
Bicgstab
Dqgmres
```

Available linear solvers can be queried with
```@docs
list_solvers
```

A default linear solver is provided for each vendor backend.
```@docs
default_linear_solver
```

A default batch linear solver is provided for each vendor backend.
```@docs
default_batch_linear_solver
```
