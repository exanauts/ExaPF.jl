# ExaPF

[ExaPF.jl](https://github.com/exanauts/ExaPF.jl) is a
package to solve powerflow problem on exascale architecture. `ExaPF.jl` aims to
implement a reduced method for solving the optimal power flow problem (OPF)
fully on GPUs. Reduced methods enforce the constraints, represented here by
the power flow's (PF) system of nonlinear equations, separately at each
iteration of the optimization in the reduced space. This paper describes the
API of `ExaPF.jl` for solving the power flow's nonlinear equations (NLE) entirely on the GPU.
This includes the computation of the derivatives using automatic
differentiation, an iterative linear solver with a preconditioner, and a
Newton-Raphson implementation. All of these steps allow us to run the main
computational loop entirely on the GPU with no transfer from host to device.

This implementation will serve as the basis for the future optimal power flow (OPF) implementation as a nonlinear programming problem (NLP)
in the reduced space.

To make our implementation portable to CPU and GPU architectures we leverage
two abstractions: arrays and kernels. Both of these abstractions are
supported through the packages [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) and [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl)
Please take a look at the [autodiff](man/autodiff.md) and [linear solver](man/linearsolver.md)
implementations to get a design overview of `ExaPF.jl` targeted for GPUs.

The user API is separated into three layers:

* First layer: Physical layer, specify the power network topology in [powersystem](man/powersystem.md)
* Second layer: Interface between power network and NLE or NLP in [formulations](lib/formulations.md)
* Third layer: Evaluators for non-linear problems

The third layer is for users working in optimization whereas the first layer is for electrical engineers. They meet in the second layer.

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
