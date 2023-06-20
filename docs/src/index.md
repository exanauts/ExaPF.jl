# ExaPF

[`ExaPF.jl`](https://github.com/exanauts/ExaPF.jl) is a
package to solve the power flow problem on upcoming exascale architectures.
The code has been designed to be:
1. **Portable:** Targeting exascale architectures implies a focus on graphics processing units (GPUs) as these systems lack substantial computational performance through classical CPUs.
1. **Differentiable:** All the expressions implemented in ExaPF are fully compatible with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl/), and routines are provided to extract first- and second-order derivatives to solve efficiently power flow and optimal power flow problems.

ExaPF implements a [vectorized modeler](man/formulations.md) for power systems, which allows
to manipulate basic expressions. All expressions are fully differentiable :
their first and second-order derivatives can be extracted efficiently
using [automatic differentiation](man/autodiff.md). In addition,
we leverage the packages [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) and
[`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) to make ExaPF portable across GPU architectures.


## Table of contents

```@contents
Pages = [
    "quickstart.md",
    "contrib.md",
]
Depth=1
```

### Manual

```@contents
Pages = [
    "man/formulations.md",
    "man/powersystem.md",
    "man/autodiff.md",
    "man/linearsolver.md",
    "man/benchmark.md",
]
Depth = 1
```

### Library

```@contents
Pages = [
    "lib/formulations.md",
    "lib/powersystem.md",
    "lib/autodiff.md",
    "lib/linearsolver.md",
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

This research was supported by the Exascale Computing Project (17-SC-20-SC),
a joint project of the U.S. Department of Energy’s Office of Science and
National Nuclear Security Administration, responsible for delivering a
capable exascale ecosystem, including software, applications, and hardware
technology, to support the nation’s exascale computing imperative.
