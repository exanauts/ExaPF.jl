# ExaPF

[ExaPF.jl](https://github.com/exanauts/ExaPF.jl) is a
package to solve the power flow problem on exascale architecture. `ExaPF.jl` aims to
provide the sensitity information required for a reduced space optimization
method for solving the optimal power flow problem (OPF)
fully on GPUs. Reduced space methods enforce the constraints, represented here by
the power flow's (PF) system of nonlinear equations, separately at each
iteration of the optimization in the reduced space. 
This includes the computation of second-order derivatives using automatic
differentiation, an iterative linear solver with a preconditioner, and a
Newton-Raphson implementation. All of these steps allow us to run the main
computational loop entirely on the GPU with no transfer from host to device.

We leverage the packages [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) and [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) to make ExaPF portable across GPU architectures.
[autodiff](man/autodiff.md) and [linear solver](man/linearsolver.md) illustrate
the design overview of `ExaPF.jl` targeted for GPUs.

The user API is separated into three layers:

1. First layer: Physical layer, specify the power network topology in [powersystem](man/powersystem.md)
2. Second layer: Interface between power network and NLE or NLP in [formulations](lib/formulations.md)
3. Third layer: Evaluators for nonlinear problems

The third layer is for numerical optimization whereas the first layer provides the physical properties at the electrical engineering level.

## Table of contents

```@contents
Pages = [
    "quickstart.md",
]
Depth=1
```

### Manual

```@contents
Pages = [
    "man/autodiff.md",
    "man/linearsolver.md",
    "man/powersystem.md",
    "man/formulations.md",
    "man/evaluators.md",
]
Depth = 1
```

### Library

```@contents
Pages = [
    "lib/autodiff.md",
    "lib/linearsolver.md",
    "lib/powersystem.md",
    "lib/formulations.md",
    "lib/evaluators.md",
]
Depth = 1
```

## Funding

This research was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.
