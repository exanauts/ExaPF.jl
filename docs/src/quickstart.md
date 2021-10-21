# Quick Start

This page introduces the first steps to set up `ExaPF.jl`.
We show how to load a power network instance and how to solve
the power flow equations both on the CPU and on the GPU.
The full script is implemented in [test/quickstart.jl](https://github.com/exanauts/ExaPF.jl/tree/master/test/quickstart.jl)

We start by importing CUDA and KernelAbstractions:
```julia
using CUDA
using KernelAbstractions
```

Then, we load ExaPF and its submodules with
```julia
using ExaPF
import ExaPF: AutoDiff
const PS = ExaPF.PowerSystem
const LS = ExaPF.LinearSolvers
```

## Short version

ExaPF loads instances from the [`pglib-opf`](https://github.com/power-grid-lib/pglib-opf)
benchmark, that may optionally be downloaded. Alternatively, ExaPF contains an artifact defined in `Artifacts.toml`
that is built from the [`ExaData`](https://github.com/exanauts/ExaData) repository containing Exascale Computing Project relevant test cases.
```julia
datafile = joinpath(artifact"ExaData", "ExaData", "case1354.m")
```
The powerflow equations can be solved in three lines of code, as
```julia
polar = ExaPF.PolarForm(datafile, CPU())
pf_algo = NewtonRaphson(; verbose=0, tol=1e-10)
convergence = ExaPF.powerflow(polar, pf_algo)
Iteration 0. Residual norm: 26.6667.
Iteration 1. Residual norm: 15.0321.
Iteration 2. Residual norm: 0.588264.
Iteration 3. Residual norm: 0.00488507.
Iteration 4. Residual norm: 1.39924e-06.
Iteration 5. Residual norm: 7.37136e-12.
```

Implicitly, ExaPF has just proceed to the following operations:
- instantiate automatically a starting point ``x_0`` from MATPOWER's data
- instantiate the Jacobian of the powerflow equations using AutoDiff.
- solve the powerflow equations iteratively, using a Newton-Raphson algorithm.

This compact syntax allows to solve quickly any powerflow equations
in a few lines a code. However, in most case, the user may want more
coarse grained control on the different objects manipulated.

## Detailed version

In what follows, we detail step by step the detailed procedure to solve
the powerflow equations.

### How to load a MATPOWER instance as a PowerNetwork object?
We start by importing a MATPOWER instance to a [`ExaPF.PowerSystem.PowerNetwork`](@ref) object:
```julia
datafile = joinpath(artifact"ExaData", "ExaData", "case9.m")
pf = PS.PowerNetwork(datafile)
```
The different fields of the object `pf` specify the characteristics
of the network. For instance, we could retrieve the number of buses
or get the indexing of the PV buses with
```julia
nbus = PS.get(pf, PS.NumberOfBuses())
pv_indexes = PS.get(pf, PS.PVIndexes())
```

However, a [`ExaPF.PowerSystem.PowerNetwork`](@ref) object stores only the **physical** attributes
of the network, independently of the mathematical formulations
we could use to model the network. To choose a particular formulation,
we need to pass the object `pf` to an [`ExaPF.AbstractFormulation`](@ref) layer.
Currently, the only layer implemented is the polar formulation,
with the [`ExaPF.PolarForm`](@ref) structure. In the future, other formulations
(e.g. `RectangularForm`) may be implemented as well.


### How to solve the powerflow equations?

To solve the powerflow equations, we need to choose a given mathematical
formulation for the equations of the network. To each formulation
corresponds a given state $x$ and control $u$.
Using polar representation of the voltage vector, such as $\bm{v} = |v|e^{j \theta}$,
each bus $i=1, \cdots, N_B$ must satisfy the power balance equations:
```math
\begin{aligned}
    p_i &= v_i \sum_{j}^{n} v_j (g_{ij}\cos{(\theta_i - \theta_j)} + b_{ij}\sin{(\theta_i - \theta_j})) \,, \\
    q_i &= v_i \sum_{j}^{n} v_j (g_{ij}\sin{(\theta_i - \theta_j)} - b_{ij}\cos{(\theta_i - \theta_j})) \,.
\end{aligned}
```
The powerflow equations
rewrite in the abstract mathematical formalism:
```math
g(x, u) = 0.
```
For a given control $u$, solving the powerflow equations resumes to find
a state $x(u)$ such that $g(x(u), u) = 0$.

To this goal, `ExaPF.jl` implements
a Newton-Raphson algorithm that allows to solve the powerflow equations
in a few lines of code.
We first instantiate a `PolarForm` object to adopt a polar formulation
as a model:
```julia
polar = ExaPF.PolarForm(pf, CPU())

```
Note that the constructor [`ExaPF.PolarForm`](@ref) takes as input a [`ExaPF.PowerSystem.PowerNetwork`](@ref) object
and a `KernelAbstractions.jl` device (here set to `CPU()` by default). We
will explain in the next section how to load a [`ExaPF.PolarForm`](@ref) object on
the GPU with the help of a `CUDADevice()`.

The Newton-Raphson solves the equation $g(x, u) = 0$ in an iterative fashion.
The algorithm solves at each step the linear equation:
```math
    x_{k+1} = - (\nabla_x g_k)^{-1} g(x_k, u).
```
Hence, the algorithm requires the following elements:

- an initial position $x_0$
- a function to solve efficiently the linear system $(\nabla_x g_k) x_{k+1} = g(x_k, u)$
- a function to evaluate the Jacobian $\nabla_x g_k$

that translate to the Julia code:
```julia
physical_state = get(polar, ExaPF.PhysicalState())
ExaPF.init_buffer!(polar, physical_state) # populate values inside buffer
linear_solver = LS.DirectSolver()
```

We build a Jacobian object storing all structures needed by
the AutoDiff backend:
```julia
julia> jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
```

Let's explain further these three objects.

- `physical_state` is a `AbstractPhysicalCache` storing all the physical values
  attached to the formulation `polar::PolarForm`.
- `jx` is a `Jacobian` structure which allows the solver to compute efficiently
  the Jacobian of the powerflow equations $\nabla_x g$ using AutoDiff.
- `linear_solver` specifies the linear algorithm uses to solve the linear
  system $(\nabla_x g_k) x_{k+1} = g(x_k, u)$. By default, we use direct linear
  algebra.

In the AutoDiff Jacobian `jx`, the evaluation of the Jacobian ``J``
is stored in `jx.J`:
```julia
jac = jx.J
```
This matrix is at the basis of the powerflow algorithm. At each
iteration, the AutoDiff backend updates the values in the Jacobian `jx`,
then we take the updated matrix `jx.J` to evaluate the

The procedure is implemented in the `powerflow` function, which
uses a Newton-Raphson algorithm to solve the powerflow equations.
The Newton-Raphson algorithm is specified as:
```julia
pf_algo = NewtonRaphson(; verbose=1, tol=1e-10)
```

Then, we could solve the powerflow equations simply with
```julia
convergence = ExaPF.powerflow(polar, jx, physical_state, pf_algo;
                              linear_solver=linear_solver)
Iteration 0. Residual norm: 26.6667.
Iteration 1. Residual norm: 15.0321.
Iteration 2. Residual norm: 0.588264.
Iteration 3. Residual norm: 0.00488507.
Iteration 4. Residual norm: 1.39924e-06.
Iteration 5. Residual norm: 7.37136e-12.
```
Here, the algorithm solves the powerflow equations in 5 iterations.
The algorithm modifies the values of `physical_state` inplace, to
avoid any unnecessary memory allocations.


### How to deport the computation on the GPU?

Now, how could we deport the resolution on the GPU?
The procedure looks exactly the same. It suffices to initiate
a new [`ExaPF.PolarForm`](@ref) object, but on the GPU:
```julia
polar_gpu = ExaPF.PolarForm(pf, CUDADevice())

```
`polar_gpu` will load all the structures it needs on the GPU, to
avoid unnecessary movements between the host and the device.
We could load the other structures directly on the GPU with:
```julia
physical_state_gpu = get(polar, ExaPF.PhysicalState())
ExaPF.init_buffer!(polar_gpu, physical_state_gpu) # populate values inside buffer
jx_gpu = AutoDiff.Jacobian(polar_gpu, ExaPF.power_balance, State())
linear_solver = DirectSolver()
```
Then, solving the powerflow equations on the GPU is straightforward
```julia
convergence = ExaPF.powerflow(polar_gpu, jx_gpu, physical_state_gpu, pf_algo;
                              linear_solver=linear_solver)
```
yielding the output
```
Iteration 0. Residual norm: 26.6667.
Iteration 1. Residual norm: 15.0321.
Iteration 2. Residual norm: 0.588264.
Iteration 3. Residual norm: 0.00488507.
Iteration 4. Residual norm: 1.39924e-06.
Iteration 5. Residual norm: 7.94916e-12.
```

Note that we get the same convergence pattern as on the CPU.


### How to solve the linear system with BICGSTAB?

By default, the algorithm runs with a direct solver, which might be
inefficient for large problems. To overcome this issue, ExaPF implements
a wrapper for different iterative algorithms (GMRES, BICGSTAB).

The performance of iterative solvers is usually improved if we use
a preconditioner.
`ExaPF.jl` implements a block-Jacobi preconditioner, tailored
for GPU usage. To build an instance with 8 blocks, just write
```julia
npartitions = 8
precond = LS.BlockJacobiPreconditioner(jac, npartitions, CUDADevice())
```
You could define an iterative solver preconditioned with `precond` simply as:
```julia
linear_solver = ExaPF.KrylovBICGSTAB(precond)

```
(this will use the BICGSTAB algorithm implemented in
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl/)).
By default, the tolerance of BICGSTAB is set to `1e-10`:
```julia
linear_solver.atol # 1e-10
```

We need to update accordingly the tolerance of the Newton-Raphson algorithm,
as it could not be lower than the tolerance of the iterative solver.
```julia
pf_algo = NewtonRaphson(; verbose=1, tol=1e-7)
```

Calling
```julia
convergence = ExaPF.powerflow(polar_gpu, jx_gpu, physical_state_gpu, pf_algo;
                              linear_solver=linear_solver)
```
yields
```
Iteration 0. Residual norm: 26.6667.
Iteration 1. Residual norm: 15.0321.
Iteration 2. Residual norm: 0.588264.
Iteration 3. Residual norm: 0.00488507.
Iteration 4. Residual norm: 1.39925e-06.
Iteration 5. Residual norm: 1.81445e-09.

```

