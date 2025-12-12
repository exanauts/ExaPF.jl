```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
    const LS = ExaPF.LinearSolvers
end
DocTestFilters = [r"ExaPF"]
```

```@setup direct_solver
using ExaPF
using KLU
using LinearAlgebra
const LS = ExaPF.LinearSolvers
```

# Direct solvers for power flow

ExaPF implements a power flow solver in the function [`run_pf`](@ref).
Under the hood, the function [`run_pf`](@ref) calls the function
[`nlsolve!`](@ref) which uses a Newton-Raphson
algorithm to solve iteratively the system of nonlinear equations
```math
g(x, p) = 0
```
where $$g: \mathbb{R}^{n_x} \times \mathbb{R}^{n_p} \to \mathbb{R}^{n_x}$$
is a nonlinear function encoding the power flow equations.

At a fixed $$p$$, solving the power flow amounts to find a
state $$x$$ such that $$g(x, p) = 0$.
At iteration $$k$$, the Newton-Raphson algorithm finds
the next iterate by solving the linear system
```math
(\nabla_x g_k) \Delta x = - g_k
```
and setting $$x_{k+1} = x_{k} + \Delta x_k$$.
The Jacobian $$\nabla_x g_k = \nabla_x (x_k, p)$$ is
computed automatically in sparse format using [AutoDiff](@ref).

Hence, solving the power flow equations amounts to solve
a sequence of sparse linear systems. When a direct solver
is employed, the system is solved in two steps. First, a
LU factorization of the matrix $$\nabla_x g$$ is computed:
we find a lower and an upper triangular matrices
$$L$$ and $$U$$ as well as two permutation matrices $$P$$ and $$Q$$
such that
```math
P (\nabla_x g) Q = LU
```
Once the matrix factorized, solving the linear system just translates
to perform two backsolves with the triangular matrices $$L$$ and $$U$$.

This method is usually efficient, as the power flow Jacobian is
super sparse (less than 1% of nonzeroes) and its sparsity pattern is fixed,
so we have to compute the symbolic factorization of the system only once.

## KLU (CPU, default)

[KLU](https://dl.acm.org/doi/abs/10.1145/1824801.1824814) is an
efficient sparse linear solver, initially designed for circuit simulation
problems.
It is often considered as one of the state-of-the-art linear solver to solve power flow problems.
Conveniently, KLU is wrapped in Julia with the package [KLU.jl](https://github.com/JuliaSparse/KLU.jl).
We use it by default when `J` is a `SparseMatrixCSC`.

Then, we are ready to solve a power flow with KLU using our current
abstraction.
```@example direct_solver
polar = ExaPF.load_polar("case9241pegase.m")
stack = ExaPF.NetworkStack(polar)
pf_solver = NewtonRaphson(tol=1e-10, verbose=2)  # power flow solver
func = ExaPF.PowerFlowBalance(polar) ∘ ExaPF.Basis(polar) # power flow func
jx = ExaPF.Jacobian(polar, func, State()) # init AD
ExaPF.nlsolve!(pf_solver, jx, stack)
```

We observe KLU reduces considerably the time spent in the linear solver.

## cuDSS (CUDA, default)

[cuDSS](https://developer.nvidia.com/cudss) is collection of direct sparse solvers implemented in CUDA.

```@example direct_solver
using CUDSS
```

We first have to instantiate everything on the GPU:

```@example direct_solver
using CUDA
polar_gpu = ExaPF.load_polar("case9241pegase.m", CUDABackend())
stack_gpu = ExaPF.NetworkStack(polar_gpu)
func_gpu = ExaPF.PowerFlowBalance(polar_gpu) ∘ ExaPF.Basis(polar_gpu)
jx_gpu = ExaPF.Jacobian(polar_gpu, func_gpu, State()) # init AD
```

Then, we are able to solve the power flow *entirely on the GPU*, simply as

```@example direct_solver
ExaPF.nlsolve!(pf_solver, jx_gpu, stack_gpu)
```
