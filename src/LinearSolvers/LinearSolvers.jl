module LinearSolvers

using LinearAlgebra
using Adapt

using Printf
using SparseArrays

import Base: show

using KernelAbstractions
import Krylov

import ..ExaPF: xnorm

import Base.size, Base.sizeof, Base.format_bytes

using KLU
import Krylov
using KrylovPreconditioners

const KA = KernelAbstractions
const KP = KrylovPreconditioners

export list_solvers, default_linear_solver, default_batch_linear_solver
export DirectSolver, Bicgstab
export do_scaling, scaling!

@enum(
    SolveStatus,
    Unsolved,
    MaxIterations,
    NotANumber,
    Converged,
    Diverged,
)

abstract type AbstractLinearSolver end
abstract type AbstractIterativeLinearSolver <: AbstractLinearSolver end

"""
    list_solvers(::KernelAbstractions.Device)

List linear solvers available on current device (CPU, NVIDIA GPU, AMD GPU).

"""
function list_solvers end

"""
    ldiv!(solver, x, A, y)
    ldiv!(solver, x, y)

* `solver::AbstractLinearSolver`: linear solver to solve the system
* `x::AbstractVector`: Solution
* `A::AbstractMatrix`: Input matrix
* `y::AbstractVector`: RHS

Solve the linear system ``A x = y`` using the algorithm
specified in `solver`. If `A` is not specified, the function
will used directly the factorization stored inside `solver`.

"""
function ldiv! end

"""
    rdiv!(solver, x, A, y)
    rdiv!(solver, x, y)

* `solver::AbstractLinearSolver`: linear solver to solve the system
* `x::AbstractVector`: Solution
* `A::AbstractMatrix`: Input matrix
* `y::AbstractVector`: RHS

Solve the linear system ``A^⊤ x = y`` using the algorithm
specified in `solver`. If `A` is not specified, the function
will used directly the factorization stored inside `solver`.

"""
function rdiv! end

_get_type(J) = error("No handling of sparse Jacobian type defined in LinearSolvers")
_get_type(J::SparseMatrixCSC) = Vector{Float64}
do_scaling(linear_solver) = false
scaling!(A,b) = nothing

"""
    DirectSolver <: AbstractLinearSolver

Solve linear system ``A x = y`` with direct linear algebra.

* On `CPU`, `DirectSolver` redirects the resolution to KLU if `A` is a `SparseMatrixCSC`.
* On CUDA GPU, `DirectSolver` redirects the resolution to cuDSS if `A` is a `CuSparseMatrixCSR`.
"""
struct DirectSolver{Fac<:LinearAlgebra.Factorization} <: AbstractLinearSolver
    factorization::Fac
end

DirectSolver(J, nbatch::Int=1; options...) = DirectSolver(klu(J))

function update!(s::DirectSolver, J::AbstractMatrix)
    klu!(s.factorization, J) # Update factorization inplace
end

# Reuse factorization in update
function ldiv!(s::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractVector, J::AbstractMatrix, x::AbstractVector; options...)
    LinearAlgebra.ldiv!(y, s.factorization, x) # Forward-backward solve
    return 0
end

# Solve system Ax = y
function ldiv!(s::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractArray, x::AbstractArray; options...)
    LinearAlgebra.ldiv!(y, s.factorization, x) # Forward-backward solve
    return 0
end

function ldiv!(s::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractArray; options...)
    LinearAlgebra.ldiv!(s.factorization, y) # Forward-backward solve
    return 0
end

# Solve system A'x = y
function rdiv!(s::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractArray, x::AbstractArray)
    LinearAlgebra.ldiv!(y, s.factorization', x) # Forward-backward solve
    return 0
end

update!(solver::AbstractIterativeLinearSolver, J::SparseMatrixCSC) = KP.update!(solver.precond, J)

"""
    Dqgmres <: AbstractIterativeLinearSolver
    Dqgmres(precond; verbose=false, memory=4)

Wrap `Krylov.jl`'s DQGMRES algorithm to solve iteratively the linear system
``A x = y``.
"""
struct Dqgmres <: AbstractIterativeLinearSolver
    inner::Krylov.DqgmresWorkspace
    precond::AbstractKrylovPreconditioner
    memory::Int
    verbose::Bool
end

function Dqgmres(J::AbstractSparseMatrix;
    P=BlockJacobiPreconditioner(J), memory=4, verbose=false
)
    n, m = size(J)
    S = _get_type(J)
    workspace = Krylov.DqgmresWorkspace(n, m, S; memory)
    return Dqgmres(workspace, P, memory, verbose)
end

function ldiv!(solver::Dqgmres,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector; options...
)
    Krylov.dqgmres!(solver.inner, J, x; N=solver.precond)
    copyto!(y, solver.inner.x)
    return Krylov.iteration_count(solver.inner)
end

"""
    Bicgstab <: AbstractIterativeLinearSolver
    Bicgstab(precond; verbose=0, rtol=1e-10, atol=1e-10)

Wrap `Krylov.jl`'s BICGSTAB algorithm to solve iteratively the linear system
``A x = y``.
"""
struct Bicgstab <: AbstractIterativeLinearSolver
    inner::Krylov.BicgstabWorkspace
    precond::AbstractKrylovPreconditioner
    verbose::Int
    atol::Float64
    rtol::Float64
    ldiv::Bool
    scaling::Bool
    maxiter::Int64
end
do_scaling(linear_solver::Bicgstab) = linear_solver.scaling
function Bicgstab(J::AbstractSparseMatrix;
    P=BlockJacobiPreconditioner(J), verbose=0, rtol=1e-10, atol=1e-10, ldiv=false, scaling=false, maxiter=size(J,1)
)
    n, m = size(J)
    S = _get_type(J)
    workspace = Krylov.BicgstabWorkspace(n, m, S)
    return Bicgstab(workspace, P, verbose, atol, rtol, ldiv, scaling, maxiter)
end

function ldiv!(solver::Bicgstab,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector;
    max_atol = solver.atol, max_rtol = solver.rtol, options...
)

    atol = max_atol < solver.atol ? solver.atol : max_atol
    rtol = max_rtol < solver.rtol ? solver.rtol : max_rtol
    Krylov.bicgstab!(
        solver.inner, J, x;
        N=solver.precond,
        atol=atol,
        rtol=rtol,
        verbose=solver.verbose,
        history=true,
        ldiv=solver.ldiv,
        itmax=solver.maxiter,
    )
    if solver.inner.stats.status == "breakdown αₖ == 0"
        @warn("BICGSTAB failed to converge. Final status is $(solver.inner.stats.status)")
    end
    copyto!(y, solver.inner.x)
    return Krylov.iteration_count(solver.inner)
end

"""
    list_solvers(::KA.CPU)

List all (batch) linear solvers available for solving the power flow on the CPU.
"""
list_solvers(::KA.CPU) = [DirectSolver, Dqgmres, Bicgstab]

"""
    default_linear_solver(A::SparseMatrixCSC, ::KA.CPU)

Default linear solver on the CPU.
"""
default_linear_solver(A::SparseMatrixCSC, device::KA.CPU) = DirectSolver(A)

"""
    default_linear_solver(A::SparseMatrixCSC, ::KA.CPU)

Default batch linear solver on the CPU.
"""
default_batch_linear_solver(A::SparseMatrixCSC, device::KA.CPU) = DirectSolver(A)

end
