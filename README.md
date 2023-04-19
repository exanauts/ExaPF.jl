# ExaPF

[![][docs-stable-img]][docs-stable-url] [![][build-latest-img]][build-url] [![][codecov-latest-img]][codecov-latest-url] [![][doi-img]][doi-url]

ExaPF is a HPC package implementing a vectorized modeler
for power systems. It targets primarily GPU architectures, and provides a portable abstraction to model power systems on upcoming HPC architectures.

Its main features are:
* **Portable approach:** All [expressions](https://exanauts.github.io/ExaPF.jl/dev/lib/formulations/#Constraints) (`PowerFlowBalance`, `CostFunction`, `PowerGenerationBounds`, ...) are evaluated fully on the GPU, without data transfers to the host.
* **Differentiable kernels:** All the expressions are differentiable with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). ExaPF uses matrix coloring to generate efficiently the Jacobian and the Hessian in sparse format.
* **Power flow solver:** ExaPF implements a power flow solver working fully on the GPU, based on a Newton-Raphson algorithm.
* **Iterative linear algebra:** ExaPF uses [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) to solve sparse linear systems entirely on the GPU, together with an overlapping Schwarz preconditioner.

ExaPF leverages [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
to generate portable kernels working on different backends.
Right now, only CUDA is fully supported, but in the medium term we have good hope to support
both [AMD ROCm](https://github.com/JuliaGPU/AMDGPU.jl) and [Intel oneAPI](https://github.com/JuliaGPU/oneAPI.jl).

## Quick-start
### How to install ExaPF?

```julia
pkg> add ExaPF
```

### Test
```julia
pkg> test ExaPF
```

### How to solve the power flow of a given MATPOWER instance?

ExaPF solves the power flow equations of a power network with a Newton-Raphson algorithm:

```julia
# Input file
case = "case57.m"
# Instantiate a PolarForm object on the CPU.
# (Replace CPU() by CUDADevice() to deport computation on a CUDA GPU)
polar = ExaPF.PolarForm(case, CPU())
# Initial variables
stack = ExaPF.NetworkStack(polar)
# Solve power flow
conv = run_pf(polar, stack; verbose=1)
```
```shell
#it 0: 6.18195e-01
#it 1: 8.19603e-03
#it 2: 7.24135e-06
#it 3: 4.68355e-12
Power flow has converged: true
  * #iterations: 3
  * Time Jacobian (s) ........: 0.0004
  * Time linear solver (s) ...: 0.0010
  * Time total (s) ...........: 0.0014
```

For more information on how to solve power flow on the GPU,
please refer to the [quickstart guide](https://exanauts.github.io/ExaPF.jl/dev/quickstart/).

## Extensions

- [Argos.jl](https://github.com/exanauts/Argos.jl/) uses ExaPF as a modeler to accelerate the resolution of OPF problems on CUDA GPU.

## Development

We welcome any contribution to ExaPF! Bug fixes or feature requests
can be reported with the [issue tracker](https://github.com/exanauts/ExaPF.jl/issues),
and new contributions can be made by opening a pull request on the `develop`
branch. For more information about development guidelines, please
refer to [CONTRIBUTING.md](https://github.com/exanauts/ExaPF.jl/blob/main/CONTRIBUTING.md)

## Funding
This research was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://exanauts.github.io/ExaPF.jl/stable

[codecov-latest-img]: https://codecov.io/gh/exanauts/ExaPF.jl/branch/main/graphs/badge.svg?branch=main
[codecov-latest-url]: https://codecov.io/github/exanauts/ExaPF.jl?branch=main

[build-url]: https://github.com/exanauts/ExaPF.jl/actions?query=workflow
[build-latest-img]: https://github.com/exanauts/ExaPF.jl/workflows/Run%20tests/badge.svg?branch=main

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.6536402.svg
[doi-url]: https://doi.org/10.5281/zenodo.6536402
