# ExaPF

[`ExaPF.jl`](https://github.com/exanauts/ExaPF.jl) is a
package to solve the power flow problem on upcoming exascale architectures by solving a system of nonlinear equations and provide derivative information used for example in a reduced space optimization method.
Targeting exascale architectures implies a focus on graphics processing units (GPUs) as these systems lack substantial computational performance through classical CPUs.
In addition to providing first-order derivatives `ExaPF.jl` includes the computation of second-order derivatives using automatic differentiation. All main computational steps, including the linear solver, are executed entirely on the GPU.
We leverage the packages [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) and [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) to make ExaPF portable across GPU architectures.
[Autodiff](man/autodiff.md) and [Linear solver](man/linearsolver.md) illustrate
the design overview of [`ExaPF.jl`](https://github.com/exanauts/ExaPF.jl) targeted for GPUs.

The user API is separated into two layers:

1. First layer: Physical layer, specify the power network topology in [PowerSystem](man/powersystem.md). The first layer provides the physical properties at the electrical engineering level.
2. Second layer: Mathematical layer, using a [Polar Formulation](lib/formulations.md) to model the equations of the network.


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
    "man/benchmark.md",
    "man/linearsolver.md",
    "man/powersystem.md",
    "man/formulations.md",
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
]
Depth = 1
```

### Artifact
```@contents
Pages = [
    "artifact.md",
]
Depth = 1
```


## Funding

This research was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.
