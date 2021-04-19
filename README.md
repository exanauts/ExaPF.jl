# ExaPF

[![][docs-latest-img]][docs-latest-url] ![CI](https://github.com/exanauts/ExaPF.jl/workflows/Run%20tests/badge.svg?branch=master) 

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://exanauts.github.io/ExaPF.jl/

ExaPF is a HPC package for solving power flow (PF) on a GPU. It currently solves PF using the Newton-Raphson algorithm on NVIDIA GPUs.
Its main features are:

* Using [CUDA.jl](https://juliagpu.gitlab.io/CUDA.jl/) CuArrays arrays for generating CUDA kernels using the broadcast '.' operator.
* Using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and Jacobian coloring to generate the compressed Jacobian of the PF equations. The Jacobian evaluation is taking place fully on the GPU.
* Preconditioned BICGSTAB with support for [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
* A block Jacobi preconditioner that updates on the GPU.

This code will serve as the basis for OPF on GPUs using the reduced gradient method. A similar abstraction than CuArrays will be used to port the code to AMD ROCm and Intel oneAPI through [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) and [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl), respectively.

## Quick-start
### Installation

```julia
pkg> add ExaPF
```

### Test
```julia
pkg> test ExaPF
```

### How to solve the power flow of a given MATPOWER instance?

ExaPF implements a Newton-Raphson algorithm to solve
the power flow equations of a power network.

```julia
# Input file
julia> case = "case57.m"
# Instantiate a PolarForm object on the CPU.
julia> polar = ExaPF.PolarForm(case, CPU())
# Instantiate a Newton-Raphson algorithm with verbose activated
julia> pf_algo = NewtonRaphson(verbose=1)
# Resolution
julia> ExaPF.powerflow(polar, algo)
Iteration 0. Residual norm: 4.295.
Iteration 1. Residual norm: 0.250361.
Iteration 2. Residual norm: 0.00441074.
Iteration 3. Residual norm: 2.81269e-06.
Iteration 4. Residual norm: 3.9111e-12.
ExaPF.ConvergenceStatus(true, 4, 3.911102241031109e-12, 0)
```

### How to solve the optimal power flow in the reduced space?

ExaPF implements a wrapper to [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl)
that allows to solve the optimal power flow problem directly in the reduced space
induced by the power flow equations:

```julia
julia> case = "case57.m"
# Instantiate a ReducedSpaceEvaluator object
julia> nlp = ExaPF.ReducedSpaceEvaluator(datafile)
# MOI optimizer
julia> optimizer = Ipopt.Optimizer()
# Use LBFGS algorithm, as reduced Hessian is not available by default!
julia> MOI.set(optimizer, MOI.RawParameter("hessian_approximation"), "limited-memory")
julia> MOI.set(optimizer, MOI.RawParameter("tol"), 1e-4)
julia> solution = ExaPF.optimize!(optimizer, nlp)
Total number of variables............................:       10
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       10
                     variables with only upper bounds:        0
Total number of equality constraints.................:        0
Total number of inequality constraints...............:       58
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:       58
        inequality constraints with only upper bounds:        0


Number of Iterations....: 9

                                   (scaled)                 (unscaled)
Objective...............:   1.9630480251946040e+03    3.7589338203438238e+04
Dual infeasibility......:   2.5545890554923290e-05    4.8916435433709606e-04
Constraint violation....:   4.7695181137896725e-13    4.7695181137896725e-13
Complementarity.........:   1.0270912626531211e-11    1.9667211572084318e-10
Overall NLP error.......:   2.5545890554923290e-05    4.8916435433709606e-04

[...]
Total CPU secs in IPOPT (w/o function evaluations)   =      0.049
Total CPU secs in NLP function evaluations           =      0.023

EXIT: Optimal Solution Found.
```

## Development

We welcome any contribution to ExaPF! Bug fixes or feature requests
can be reported with the [issue tracker](https://github.com/exanauts/ExaPF.jl/issues),
and new contributions can be made by opening a pull request on the `develop`
branch. For more information about development guidelines, please
refer to [CONTRIBUTING.md](https://github.com/exanauts/ExaPF.jl/blob/master/CONTRIBUTING.md)

## Funding
This research was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.

