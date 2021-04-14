
```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
end
DocTestFilters = [r"ExaPF"]
```

# Benchmark

For the purpose of performance regression testing ExaPF provides a lightweight benchmark script to test the various configurations for the linear solvers used in the Newton-Raphson algorithm and run them on a specific hardware. The main julia script [benchmark/benchmarks.jl](https://github.com/exanauts/ExaPF.jl/tree/master/benchmark/benchmarks.jl) takes all options from the command line. This the linear solver (e.g. `KrylovBICGSTAB`), the architecture as a `KernelAbstractions` object (CPU or CUDADevice), and the case filename in (e.g. `data/case9.m`. An exhaustive list of all available linear solvers can be obtained via [`ExaPF.LinearSolvers.list_solvers`](@ref).

Running 
```
julia --project benchmark/benchmarks.jl KrylovBICGSTAB CUDADevice case300.m
```
yields
```
KrylovBICGSTAB, CUDADevice, case300.m,  69.0,  3.57,  43.7, true
```
The first three fields are the settings of the benchmark run. They are followed by three timings in milliseconds:
* the total Newton-Raphson algorithm for solving the power flow,
* the timings for the Jacobian accumulation using [AutoDiff](autodiff.md),
* and the time for the linear solver, including the preconditioner.
To acquire these timings the code is run three times to avoid any precompilation effects. The last field confirms the Newton-Raphson convergence. In case more verbose output is desired, one has to manually set the verbosity in [benchmark/benchmarks.jl](https://github.com/exanauts/ExaPF.jl/tree/master/benchmark/benchmarks.jl) by changing
```julia
powerflow_solver = NewtonRaphson(tol=ntol)
```
to one of the following options:
```julia
powerflow_solver = NewtonRaphson(tol=ntol, verbose=VERBOSE_LEVEL_NONE)
powerflow_solver = NewtonRaphson(tol=ntol, verbose=VERBOSE_LEVEL_LOW)
powerflow_solver = NewtonRaphson(tol=ntol, verbose=VERBOSE_LEVEL_MEDIUM)
powerflow_solver = NewtonRaphson(tol=ntol, verbose=VERBOSE_LEVEL_HIGH)
```
A shell script [benchmark/benchmarks.sh](https://github.com/exanauts/ExaPF.jl/tree/master/benchmark/benchmarks.sh) is provided to gather timings with various canonical configurations and storing them in a file `cpu_REV.log` and `gpu_REF.log`, where `REV` is the sha1 hash of the current checked out ExaPF version.
