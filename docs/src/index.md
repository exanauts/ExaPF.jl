# ExaPF

[`ExaPF.jl`](https://github.com/exanauts/ExaPF.jl) is a
package to solve the power flow problem on upcoming exascale architectures.
On these architectures the computational performance can only be achieved through graphics processing units (GPUs) as these systems lack substantial computational performance through classical CPUs.
[`ExaPF.jl`](https://github.com/exanauts/ExaPF.jl) aims to
provide the sensitivity information required for a reduced space optimization
method, and enabling the computation of the optimal power flow problem (OPF)
fully on GPUs. Reduced space methods enforce the constraints, represented here by
the power flow's (PF) system of nonlinear equations, separately at each
iteration of the optimization in the reduced space.
This includes the computation of second-order derivatives using automatic
differentiation, an iterative linear solver with a preconditioner, and a
Newton-Raphson implementation. All of these steps allow us to run the main
computational loop entirely on the GPU with no transfer from host to device.

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
