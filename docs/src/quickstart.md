# Quick Start

This page introduces the first steps to set up `ExaPF.jl`.
We show how to load a power network instance and how to solve
the power flow equations both on the CPU and on the GPU.

### How to load a MATPOWER instance?
We start by importing into `ExaPF` an instance specified in the MATPOWER format.

First, you could load the package with
```julia-repl
julia> using ExaPF
julia> const PS = ExaPF.PowerSystem
```

Imagine you want to load an instance from the [`pglib-opf`](https://github.com/power-grid-lib/pglib-opf)
benchmark, stored in the current folder:
```julia-repl
julia> pglib_instance = "pglib_opf_case1354_pegase.m"
```
`ExaPF.jl` allows you to load directly the instance as a `PowerNetwork`
object:
```julia-repl
julia> pf = PS.PowerNetwork(pglib_instance)
```
The different fields of the object `pf` specify the characteristics
of the network. For instance, we could retrieve the number of buses
or get the indexing of the PV buses with
```julia-repl
julia> nbus = pf.nbus
1354
julia> pv_indexes = PS.get(pf, PS.PVIndexes())
[17, 21, ..., 1344]
```

However, a `PowerNetwork` object stores only the **physical** attributes
of the network, independently from the mathematical formulations
we could use to model the network. To choose a particular formulation,
we need to pass the object `pf` to a `AbstractFormulation` layer.
Currently, the only layer implemented is the polar formulation,
with the `PolarForm` abstraction. In the future, other formulations
(e.g. `RectangularForm`) may be implemented as well.


### How to solve the powerflow equations?

To solve the powerflow equations, we need to choose a given mathematical
formulation for the equations of the network. To each formulation
corresponds a given state $x$ and control $u$. The powerflow equations
write in the abstract mathematical formalism:
```math
g(x, u) = 0.
```
For a given control $u$, solving the powerflow equations resumes to find
a state $x(u)$ such that $g(x(u), u) = 0$. To this goal, `ExaPF.jl` implements
a Newton-Raphson algorithm.

We first instantiate a `PolarForm` object to adopt a polar formulation
as a model:
```julia-repl
julia> polar = PolarForm(pf, CPU())

```
Note that the constructor `PolarForm` takes as input a `PowerNetwork` object
and a `KernelAbstractions.jl` device (here set to `CPU()` by default). We
will explain in the next section how to load a `PolarForm` object on
the GPU with the help of a `CUDADevice`.

The Newton-Raphson solves the equation $g(x, u) = 0$ in an iterative fashion.
The algorithm solves at each step the linear equation:
```math
    x_{k+1} = - (\nabla_x g_k)^{-1} g(x_k, u).
```
Hence, the algorithm requires the following elements:

- an initial position $x_0$
- a function to evaluate the Jacobian $\nabla_x g_k$
- a function to solve efficiently the linear system $(\nabla_x g_k) x_{k+1} = g(x_k, u)$

that translate to the Julia code:
```julia-repl
julia> physical_state = get(polar, PhysicalState())
julia> jx = ExaPF.init_ad_factory(polar, physical_state)
julia> linear_solver = DirectSolver()

```
Let's explain further these three objects.

- `physical_state` is a `AbstractPhysicalCache` storing all the physical values
  attached to the formulation `polar::PolarForm`.
- `jx` is a `StateJacobianAD` which allows the solver to compute efficiently
  by automatic differentiation the Jacobian of the powerflow equations $\nabla_x g$.
- `linear_solver` specifies the linear algorithm uses to solve the linear
  system $(\nabla_x g_k) x_{k+1} = g(x_k, u)$. By default, we use direct linear
  algebra.


Then, we could solve the powerflow equations simply with
```julia-repl
julia> convergence = ExaPF.powerflow(polar, jx, physical_state,
                                     linear_solver=linear_solver,
                                     verbose_level=1)
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

Now, what if we want to deport the resolution on the GPU?
The procedure looks exactly the same. It suffices to initiate
a new `PolarForm` object, but on the GPU:
```julia-repl
julia> polar_gpu = PolarForm(pf, CUDADevice())

```
`polar_gpu` will load all the structure it needs on the GPU, to
avoid unnecessary movements between the host and the GPU.
We could load the other structures directly on the GPU with:
```julia-repl
julia> physical_state_gpu = get(polar, PhysicalState())
julia> jx_gpu = ExaPF.init_ad_factory(polar, physical_state)
julia> linear_solver = DirectSolver()

```
and then, solving the powerflow equations on the GPU is
straightforward:
```julia-repl
julia> convergence = ExaPF.powerflow(polar_gpu, jx_gpu, physical_state_gpu,
                                     linear_solver=linear_solver,
                                     verbose_level=1)
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
inefficient for large problems. We might want to use an iterative
solver instead. `ExaPF.jl` allows to use a block-Jacobi preconditioner:
```julia-repl
julia> const LS = ExaPF.LinearSolvers
julia> npartitions = 8
julia> precond = LS.BlockJacobiPreconditioner(jac, npartitions, CUDADevice())
```
The iterative linear solver is instantiated with:
```julia-repl
julia> linear_solver = ExaPF.BICGSTAB(precond)

```
By default, the tolerance of BICGSTAB is set to `1e-8`:
```julia-repl
julia> linear_solver.tol
1e-8
```

We need to update accordingly the tolerance of the Newton-Raphson algorithm,
as it could not be lower than the tolerance of the iterative solver.
```julia-repl
julia> convergence = ExaPF.powerflow(polar_gpu, jx_gpu, physical_state_gpu,
                                     linear_solver=linear_solver,
                                     tol=1e-7,
                                     verbose_level=1)
Iteration 0. Residual norm: 26.6667.
Iteration 1. Residual norm: 15.0321.
Iteration 2. Residual norm: 0.588264.
Iteration 3. Residual norm: 0.00488507.
Iteration 4. Residual norm: 1.39925e-06.
Iteration 5. Residual norm: 1.81445e-09.

```

