module LinearSolvers

using LinearAlgebra

using Printf
using SparseArrays

import Base: show

using CUDA
using KernelAbstractions
import CUDA.CUBLAS
import CUDA.CUSOLVER
import CUDA.CUSPARSE
import Krylov
import LightGraphs
import Metis

import ..ExaPF: xnorm

const KA = KernelAbstractions


export bicgstab, list_solvers
export DirectSolver, BICGSTAB, EigenBICGSTAB, KrylovBICGSTAB

@enum(
    SolveStatus,
    Unsolved,
    MaxIterations,
    NotANumber,
    Converged,
    Diverged,
)

include("preconditioners.jl")
include("bicgstab.jl")
include("bicgstab_eigen.jl")

abstract type AbstractLinearSolver end
abstract type AbstractIterativeLinearSolver <: AbstractLinearSolver end

"""
    list_solvers(::KernelAbstractions.Device)

List linear solvers available on current device.

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

"""
    DirectSolver <: AbstractLinearSolver

Solve linear system ``A x = y`` with direct linear algebra.

* On the CPU, `DirectSolver` uses UMFPACK to solve the linear system
* On CUDA GPU, `DirectSolver` redirects the resolution to the method `CUSOLVER.csrlsvqr`

"""
struct DirectSolver{Fac<:Union{Nothing, LinearAlgebra.Factorization}} <: AbstractLinearSolver
    factorization::Fac
end

DirectSolver(J; options...) = DirectSolver(lu(J))
DirectSolver(J::CUSPARSE.CuSparseMatrixCSR; options...) = DirectSolver(nothing)
DirectSolver() = DirectSolver(nothing)

function update!(s::DirectSolver, J::AbstractMatrix)
    lu!(s.factorization, J) # Update factorization inplace
end

# Reuse factorization in update
function ldiv!(s::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractVector, J::AbstractMatrix, x::AbstractVector)
    LinearAlgebra.ldiv!(y, s.factorization, x) # Forward-backward solve
    return 0
end
# Solve system Ax = y
function ldiv!(s::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractArray, x::AbstractArray)
    LinearAlgebra.ldiv!(y, s.factorization, x) # Forward-backward solve
    return 0
end
function ldiv!(s::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractArray)
    LinearAlgebra.ldiv!(s.factorization, y) # Forward-backward solve
    return 0
end
# Solve system A'x = y
function rdiv!(s::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractArray, x::AbstractArray)
    LinearAlgebra.ldiv!(y, s.factorization', x) # Forward-backward solve
    return 0
end

function ldiv!(::DirectSolver{Nothing}, y::Vector, J::AbstractMatrix, x::Vector)
    F = lu(J)
    LinearAlgebra.ldiv!(y, F, x)
    return 0
end

function update!(solver::AbstractIterativeLinearSolver, J::SparseMatrixCSC)
    update(solver.precond, J, CPU())
end
function update!(solver::AbstractIterativeLinearSolver, J::CUSPARSE.CuSparseMatrixCSR)
    update(solver.precond, J, CUDABackend())
end

"""
    BICGSTAB <: AbstractIterativeLinearSolver
    BICGSTAB(precond; maxiter=2_000, tol=1e-8, verbose=false)

Custom BICGSTAB implementation to solve iteratively the linear system
``A  x = y``.
"""
struct BICGSTAB <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    maxiter::Int
    tol::Float64
    verbose::Bool
end
function BICGSTAB(J::AbstractSparseMatrix;
    P=BlockJacobiPreconditioner(J), maxiter=2_000, tol=1e-8, verbose=false
)
    return BICGSTAB(P, maxiter, tol, verbose)
end

function ldiv!(solver::BICGSTAB,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    y[:], n_iters, status = bicgstab(J, x, solver.precond, y; maxiter=solver.maxiter,
                                     verbose=solver.verbose, tol=solver.tol)
    if status != Converged
        @warn("BICGSTAB failed to converge. Final status is $(status)")
    end
    return n_iters
end

"""
    EigenBICGSTAB <: AbstractIterativeLinearSolver
    EigenBICGSTAB(precond; maxiter=2_000, tol=1e-8, verbose=false)

Julia's port of Eigen's BICGSTAB to solve iteratively the linear system
``A x = y``.
"""
struct EigenBICGSTAB <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    maxiter::Int
    tol::Float64
    verbose::Bool
end
function EigenBICGSTAB(J::AbstractSparseMatrix;
    P=BlockJacobiPreconditioner(J), maxiter=2_000, tol=1e-8, verbose=false
)
    return EigenBICGSTAB(P, maxiter, tol, verbose)
end

function ldiv!(solver::EigenBICGSTAB,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    y[:], n_iters, status = bicgstab_eigen(J, x, solver.precond, y; maxiter=solver.maxiter,
                                           verbose=solver.verbose, tol=solver.tol)
    if status != Converged
        @warn("EigenBICGSTAB failed to converge. Final status is $(status)")
    end

    return n_iters
end

"""
    DQGMRES <: AbstractIterativeLinearSolver
    DQGMRES(precond; verbose=false, memory=4)

Wrap `Krylov.jl`'s DQGMRES algorithm to solve iteratively the linear system
``A x = y``.
"""
struct DQGMRES <: AbstractIterativeLinearSolver
    inner::Krylov.DqgmresSolver
    precond::AbstractPreconditioner
    memory::Int
    verbose::Bool
end
function DQGMRES(J::AbstractSparseMatrix;
    P=BlockJacobiPreconditioner(J), memory=4, verbose=false
)
    n, m = size(J)
    S = isa(J, CUSPARSE.CuSparseMatrixCSR) ? CuArray{Float64, 1, CUDA.Mem.DeviceBuffer} : Vector{Float64}
    solver = Krylov.DqgmresSolver(n, m, memory, S)
    return DQGMRES(solver, P, memory, verbose)
end

function ldiv!(solver::DQGMRES,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    CUDA.allowscalar() do
        Krylov.dqgmres!(solver.inner, J, x; N=solver.precond)
    end
    copyto!(y, solver.inner.x)
    return length(solver.inner.stats.residuals)
end

"""
    KrylovBICGSTAB <: AbstractIterativeLinearSolver
    KrylovBICGSTAB(precond; verbose=0, rtol=1e-10, atol=1e-10)

Wrap `Krylov.jl`'s BICGSTAB algorithm to solve iteratively the linear system
``A x = y``.
"""
struct KrylovBICGSTAB <: AbstractIterativeLinearSolver
    inner::Krylov.BicgstabSolver
    precond::AbstractPreconditioner
    verbose::Int
    atol::Float64
    rtol::Float64
end
function KrylovBICGSTAB(J::AbstractSparseMatrix;
    P=BlockJacobiPreconditioner(J), verbose=0, rtol=1e-10, atol=1e-10
)
    n, m = size(J)
    S = isa(J, CUSPARSE.CuSparseMatrixCSR) ? CuArray{Float64, 1, CUDA.Mem.DeviceBuffer} : Vector{Float64}
    solver = Krylov.BicgstabSolver(n, m, S)
    return KrylovBICGSTAB(solver, P, verbose, atol, rtol)
end

function ldiv!(solver::KrylovBICGSTAB,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    CUDA.allowscalar() do
        Krylov.bicgstab!(
            solver.inner, J, x;
            N=solver.precond,
            atol=solver.atol,
            rtol=solver.rtol,
            verbose=solver.verbose,
            history=true,
        )
    end
    copyto!(y, solver.inner.x)
    return length(solver.inner.stats.residuals)
end

"""
    list_solvers(::KA.CPU)

List all linear solvers available solving the power flow on the CPU.
"""
list_solvers(::KA.CPU) = [DirectSolver, DQGMRES, BICGSTAB, EigenBICGSTAB, KrylovBICGSTAB]

"""
    list_solvers(::KA.GPU)

List all linear solvers available solving the power flow on an NVIDIA GPU.
"""
list_solvers(::KA.GPU) = [DirectSolver, BICGSTAB, DQGMRES, EigenBICGSTAB, KrylovBICGSTAB]
end
