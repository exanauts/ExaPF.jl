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

## UMFPACK (CPU)

As a generic fallback, ExaPF employs the linear solver [UMFPACK](https://people.sc.fsu.edu/~jburkardt/f77_src/umfpack/umfpack.html)
to solve the linear system, as UMFPACK is shipped automatically in Julia.

In the `LinearSolvers` submodule, this is how the wrapper is implemented:
```julia
struct DirectSolver{Fac} <: AbstractLinearSolver
    factorization::Fac
end
DirectSolver(J::AbstractMatrix) = DirectSolver(lu(J))

```
By default, the constructor takes as input the initial Jacobian `J` and
factorizes it by calling `lu(J)`, which in Julia translates to a factorization
with UMFPACK. Then, inside the function [`nlsolve!`](@ref) we refactorize
the matrix at each iteration by calling the function `LinearSolvers.update!`
```julia
function update!(s::DirectSolver, J::AbstractMatrix)
    LinearAlgebra.lu!(s.factorization, J) # Update factorization inplace
end
```
This function uses the function `LinearAlgebra.lu!` to update the factorization inplace.
The backsolve is computed by calling the `LinearAlgebra.ldiv!` function:
```julia
function ldiv!(s::DirectSolver, y::AbstractVector, J::AbstractMatrix, x::AbstractVector)
    LinearAlgebra.ldiv!(y, s.factorization, x) # Forward-backward solve
    return 0
end
```
We notice that the code has been designed to support any factorization
routines implementing the two routines `LinearAlgebra.lu!` and `LinearAlgebra.ldiv!`.

Before comparing with other linear solvers, we solve a large scale
power flow instance with UMFPACK to give us a reference.
```@example direct_solver
polar = ExaPF.load_polar("case9241pegase.m")
stack = ExaPF.NetworkStack(polar)
pf_solver = NewtonRaphson(tol=1e-10, verbose=2)  # power flow solver
func = ExaPF.PowerFlowBalance(polar) ∘ ExaPF.PolarBasis(polar) # power flow func
jx = ExaPF.Jacobian(polar, func, State()) # init AD
ExaPF.nlsolve!(pf_solver, jx, stack)
```

## KLU (CPU, default)

[KLU](https://dl.acm.org/doi/abs/10.1145/1824801.1824814) is an
efficient sparse linear solver, initially designed for circuit simulation
problems.
It is often considered as one of the state-of-the-art linear solver
to solve power flow problems.
Conveniently, KLU is wrapped in Julia
with the package [KLU.jl](https://github.com/JuliaSparse/KLU.jl).
KLU.jl implements a proper interface to use KLU. We just have to implement a forgiving function for `LinearAlgebra.lu!`
```@example direct_solver
LinearAlgebra.lu!(K::KLU.KLUFactorization, J) = KLU.klu!(K, J)
```
We use by default when `J` is a `SparseMatrixCSC`.
Then, we are ready to solve a power flow with KLU using our current
abstraction. One has just to create a new instance of [`LS.DirectSolver`](@ref):
```@example direct_solver
klu_factorization = KLU.klu(jx.J)
klu_solver = LS.DirectSolver(klu_factorization)
```

and pass it to [`nlsolve!`](@ref):
```@example direct_solver
ExaPF.init!(polar, stack) # reinit stack
ExaPF.nlsolve!(pf_solver, jx, stack; linear_solver=klu_solver)
```

We observe KLU reduces considerably the time spent in the linear solver.


## cuDSS (CUDA)

[cuDSS](https://developer.nvidia.com/cudss)
is an efficient LU refactorization routine implemented in CUDA.
```@example direct_solver
using CUDSS
```

The principle is the following: the initial symbolic factorization
is computed on the CPU with the routine chosen by the user. Then,
each time we have to refactorize a matrix **with the same sparsity pattern**,
we can recompute the numerical factorization entirely on the GPU.
In practice, this solver is efficient at refactorizing a given matrix
if the sparsity is significant.

This is of direct relevance for us, as (i) the sparsity of the power
flow Jacobian doesn't change along the Newton iterations and
(ii) the Jacobian is super-sparse. In ExaPF, it is the linear solver
of choice when it comes to solve the power flow entirely on the GPU.

We first have to instantiate everything on the GPU:
```@example direct_solver
using CUDA
using CUDSS
polar_gpu = ExaPF.load_polar("case9241pegase.m", CUDABackend())
stack_gpu = ExaPF.NetworkStack(polar_gpu)
func_gpu = ExaPF.PowerFlowBalance(polar_gpu) ∘ ExaPF.PolarBasis(polar_gpu)
jx_gpu = ExaPF.Jacobian(polar_gpu, func_gpu, State()) # init AD
```
We can instantiate a new cuDSS's instance as
```@example direct_solver
rf_fac = CUDSS.lu(jx_gpu.J)
rf_solver = LS.DirectSolver(rf_fac)

```
Then, we are able to solve the power flow *entirely on the GPU*, simply as
```@example direct_solver
ExaPF.nlsolve!(pf_solver, jx_gpu, stack_gpu; linear_solver=rf_solver)

```

## cusolverRF (CUDA)

[cusolverRF](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverRF-reference)
is an efficient LU refactorization routine implemented in CUDA.
It is wrapped in Julia inside the package [CUSOLVERRF.jl](https://github.com/exanauts/CUSOLVERRF.jl):
```@example direct_solver
using CUSOLVERRF
```

The principle is the following: the initial symbolic factorization
is computed on the CPU with the routine chosen by the user. Then,
each time we have to refactorize a matrix **with the same sparsity pattern**,
we can recompute the numerical factorization entirely on the GPU.
In practice, this solver is efficient at refactorizing a given matrix
if the sparsity is significant.

This is of direct relevance for us, as (i) the sparsity of the power
flow Jacobian doesn't change along the Newton iterations and
(ii) the Jacobian is super-sparse. In ExaPF, it is the linear solver
of choice when it comes to solve the power flow entirely on the GPU.

CUSOLVERRF.jl follows the LinearAlgebra's interface, so we can use it directly in ExaPF.
We first have to instantiate everything on the GPU:
```@example direct_solver
using CUDA
polar_gpu = ExaPF.load_polar("case9241pegase.m", CUDABackend())
stack_gpu = ExaPF.NetworkStack(polar_gpu)
func_gpu = ExaPF.PowerFlowBalance(polar_gpu) ∘ ExaPF.PolarBasis(polar_gpu)
jx_gpu = ExaPF.Jacobian(polar_gpu, func_gpu, State()) # init AD
```
We can instantiate a new cusolverRF's instance as
```@example direct_solver
rf_fac = CUSOLVERRF.RFLU(jx_gpu.J)
rf_solver = LS.DirectSolver(rf_fac)
```
Then, we are able to solve the power flow *entirely on the GPU*, simply as
```@example direct_solver
ExaPF.nlsolve!(pf_solver, jx_gpu, stack_gpu; linear_solver=rf_solver)
```

## cuDSS (CUDA, default)

[cuDSS](https://developer.nvidia.com/cudss)
is collection of sparse direct solvers implemented in CUDA.

```@example direct_solver
using CUDSS
```

We first have to instantiate everything on the GPU:

```@example direct_solver
using CUDA
polar_gpu = ExaPF.load_polar("case9241pegase.m", CUDABackend())
stack_gpu = ExaPF.NetworkStack(polar_gpu)
func_gpu = ExaPF.PowerFlowBalance(polar_gpu) ∘ ExaPF.PolarBasis(polar_gpu)
jx_gpu = ExaPF.Jacobian(polar_gpu, func_gpu, State()) # init AD
```

We can instantiate a new cuDSS's solver as

```@example direct_solver
cudss_fac = CUDSS.lu(jx_gpu.J)
cudss_solver = LS.DirectSolver(cudss_fac)
```

Then, we are able to solve the power flow *entirely on the GPU*, simply as

```@example direct_solver
ExaPF.nlsolve!(pf_solver, jx_gpu, stack_gpu; linear_solver=cudss_solver)
```
