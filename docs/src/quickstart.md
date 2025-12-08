# Quick Start

This page introduces the first steps to set up `ExaPF.jl`.
We show how to load a power network instance and how to solve
the power flow equations both on the CPU and on the GPU.
The full script is implemented in [test/quickstart.jl](https://github.com/exanauts/ExaPF.jl/tree/main/test/quickstart.jl).

We start by importing CUDA and KernelAbstractions:
```@julia
using CUDA
using KernelAbstractions
```

Then, we load ExaPF and its submodules with
```@julia
using ExaPF
import ExaPF: AutoDiff
const PS = ExaPF.PowerSystem
const LS = ExaPF.LinearSolvers
```

## Short version

ExaPF loads instances from the [`pglib-opf`](https://github.com/power-grid-lib/pglib-opf)
benchmark. ExaPF contains an artifact defined in `Artifacts.toml`
that is built from the [`ExaData`](https://github.com/exanauts/ExaData) repository containing Exascale Computing Project relevant test cases. You may set a data file using
```julia
datafile = joinpath(artifact"ExaData", "ExaData", "case1354.m")
```

```@setup quickstart
using LazyArtifacts
using ExaPF
using CUDA
using KernelAbstractions
using ExaPF
import ExaPF: AutoDiff
const PS = ExaPF.PowerSystem
const LS = ExaPF.LinearSolvers
artifact_toml = joinpath(@__DIR__, "..", "..", "Artifacts.toml")
exadata_hash = artifact_hash("ExaData", artifact_toml)
datafile = joinpath(artifact_path(exadata_hash), "ExaData", "case1354.m")
```
The powerflow equations can be solved in three lines of code, as
```@repl quickstart
polar = ExaPF.PolarForm(datafile, CPU())  # Load data
stack = ExaPF.NetworkStack(polar)         # Load variables
convergence = run_pf(polar, stack; verbose=1)
```

Implicitly, ExaPF has just proceed to the following operations:
- instantiate automatically a starting point ``x_0`` from the variables stored in `stack`
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
```@repl quickstart
pf = PS.PowerNetwork(datafile)
```
The different fields of the object `pf` specify the characteristics
of the network. For instance, we can retrieve the number of buses
or get the indexing of the PV buses with
```@repl quickstart
nbus = PS.get(pf, PS.NumberOfBuses())
pv_indexes = pf.pv;
```

However, a [`ExaPF.PowerSystem.PowerNetwork`](@ref) object stores only the **physical** attributes
of the network.
To choose a given mathematical formulation,
we need to pass the object `pf` to an [`ExaPF.AbstractFormulation`](@ref) layer.
Currently, only the polar formulation is provided
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
```@repl quickstart
polar = ExaPF.PolarForm(pf, CPU())
```
Note that the constructor [`ExaPF.PolarForm`](@ref) takes as input a [`ExaPF.PowerSystem.PowerNetwork`](@ref) object
and a `KernelAbstractions.jl` backend (here set to `CPU()` by default). We
will explain in the next section how to load a [`ExaPF.PolarForm`](@ref) object on
the GPU with the help of a `CUDABackend()`.

The Newton-Raphson solves the equation $g(x, u) = 0$ in an iterative fashion.
The algorithm solves at each step the linear equation:
```math
    x_{k+1} = - (\nabla_x g_k)^{-1} g(x_k, u).
```
Hence, the algorithm requires the following elements:

- an initial variable $x_0$
- a function to solve efficiently the linear system $(\nabla_x g_k) x_{k+1} = g(x_k, u)$
- a function to evaluate the Jacobian $\nabla_x g_k$

The variable $x$ is instantiated as:
```@repl quickstart
stack = ExaPF.NetworkStack(polar)
```
The function $g$ is implemented using ExaPF's custom modeler:
```@repl quickstart
basis = ExaPF.PolarBasis(polar)
powerflow = ExaPF.PowerFlowBalance(polar) ∘ basis
```

The Jacobian $\nabla_x g$ is evaluated automatically using
forward-mode AutoDiff:
```@repl quickstart
mapx = ExaPF.mapping(polar, State());
jx = ExaPF.Jacobian(polar, powerflow, mapx)
```
The (direct) linear solver can be instantiated directly as
```@repl quickstart
linear_solver = LS.DirectSolver(jx.J);
```
Let's explain further these three objects.

- `stack` is a `AbstractStack` storing all the variables
  attached to the formulation `polar::PolarForm`.
- `jx` is a `Jacobian` structure which allows the solver to compute efficiently
  the Jacobian of the powerflow equations $\nabla_x g$ using AutoDiff.
- `linear_solver` specifies the linear algorithm uses to solve the linear
  system $(\nabla_x g_k) x_{k+1} = g(x_k, u)$. By default, we use direct sparse linear solvers.

In the AutoDiff Jacobian `jx`, the evaluation of the Jacobian ``J``
is stored in `jx.J`:
```@repl quickstart
jac = jx.J
```
This matrix is at the basis of the powerflow algorithm. At each
iteration, the AutoDiff backend updates the nonzero values in the sparse Jacobian `jx`
and solve the associated linear system to compute the next descent direction.

The procedure is implemented in the `nlsolve!` function, which
uses a Newton-Raphson algorithm to solve the powerflow equations.
The Newton-Raphson algorithm is specified as:
```@repl quickstart
pf_algo = NewtonRaphson(; verbose=1, tol=1e-10)
```

Then, we can solve the powerflow equations simply with
```@repl quickstart
convergence = ExaPF.nlsolve!(pf_algo, jx, stack; linear_solver=linear_solver)
```
Here, the algorithm solves the powerflow equations in 5 iterations.
The algorithm modifies the values of `stack` inplace, to
avoid any unnecessary memory allocations.


### How to deport the computation on the GPU?

Now, how can we deport the resolution on the GPU?
The procedure looks exactly the same. It suffices to initiate
a new [`ExaPF.PolarForm`](@ref) object, but on the GPU:
```@repl quickstart
polar_gpu = ExaPF.PolarForm(pf, CUDABackend())
```

`polar_gpu` will load all the structures it needs on the GPU, to
avoid unnecessary movements between the host and the backend.
We can load the other structures directly on the GPU with:
```@repl quickstart
stack_gpu = ExaPF.NetworkStack(polar_gpu)

basis_gpu = ExaPF.PolarBasis(polar_gpu)
pflow_gpu = ExaPF.PowerFlowBalance(polar_gpu) ∘ basis_gpu
jx_gpu = ExaPF.Jacobian(polar_gpu, pflow_gpu, mapx)

linear_solver = LS.DirectSolver(jx_gpu.J)
```

Then, solving the powerflow equations on the GPU directly
translates as
```@repl quickstart
convergence = ExaPF.nlsolve!(pf_algo, jx_gpu, stack_gpu; linear_solver=linear_solver)
```

Note that we get exactly the same iterations as when we solve the power
flow equations on the CPU.


### How to solve the linear system with BICGSTAB?

By default, the algorithm runs with a direct solver, which might be
inefficient for large problems. To overcome this issue, ExaPF implements
a wrapper for different iterative algorithms (GMRES, BICGSTAB).

The performance of iterative solvers is usually improved if we use
a preconditioner.
`ExaPF.jl` implements an overlapping Schwarz preconditioner, tailored
for GPU usage. To build an instance with 8 blocks, just write
```@repl quickstart
import KrylovPreconditioners as KP

npartitions = 8;
jac_gpu = jx_gpu.J;
precond = KP.BlockJacobiPreconditioner(jac_gpu, npartitions, CUDABackend());
```
You can attach the preconditioner to an BICGSTAB algorithm simply as
```@repl quickstart
linear_solver = ExaPF.Bicgstab(jac_gpu; P=precond);

```
(this will use the BICGSTAB algorithm implemented in
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl/)).

We need to update accordingly the tolerance of the Newton-Raphson algorithm
(the iterative solver is less accurate than the direct solver):
```@repl quickstart
pf_algo = NewtonRaphson(; verbose=1, tol=1e-7)
```

We reset the variables to their initial values:
```@repl quickstart
ExaPF.init!(polar_gpu, stack_gpu)
```
Then, solving the power flow with the iterative solvers
directly translates to one call to `nlsolve!`:
```@repl quickstart
convergence = ExaPF.nlsolve!(pf_algo, jx_gpu, stack_gpu; linear_solver=linear_solver)
```
