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

import ..ExaPF: xnorm, csclsvqr!

const KA = KernelAbstractions


export bicgstab, list_solvers
export DirectSolver, BICGSTAB, EigenBICGSTAB, KrylovBICGSTAB
export get_transpose

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

get_transpose(::AbstractLinearSolver, M::AbstractMatrix) = transpose(M)

"""
    ldiv!(solver, x, A, y)
    ldiv!(solver, x, y)

* `solver::AbstractLinearSolver`: linear solver to solve the system
* `x::AbstractVector`: Solution
* `A::AbstractMatrix`: Input matrix
* `y::AbstractVector`: RHS

Solve the linear system ``A x = y`` using the algorithm
specified in `solver`. If `A` is not specified, the function
will used directly the factorization stored inplace.

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
will used directly the factorization stored inplace.

"""
function rdiv! end

"""
    DirectSolver <: AbstractLinearSolver

Solve linear system ``A x = y`` with direct linear algebra.

* On the CPU, `DirectSolver` uses UMFPACK to solve the linear system
* On CUDA GPU, `DirectSolver` redirects the resolution to the method `CUSOLVER.csrlsvqr`

"""
struct DirectSolver{Fac} <: AbstractLinearSolver
    factorization::Fac
end

DirectSolver(J::SparseMatrixCSC{T, Int}; options...) where T = DirectSolver(lu(J))

# Reuse factorization in update
function ldiv!(s::DirectSolver, y::AbstractVector, J::AbstractMatrix, x::AbstractVector)
    lu!(s.factorization, J) # Update factorization inplace
    LinearAlgebra.ldiv!(y, s.factorization, x)
    return 0
end
function rdiv!(s::DirectSolver, y::AbstractVector, J::AbstractMatrix, x::AbstractVector)
    lu!(s.factorization, J) # Update factorization inplace
    LinearAlgebra.ldiv!(y, s.factorization', x)
    return 0
end
# Solve system Ax = y
function ldiv!(s::DirectSolver, y::AbstractArray, x::AbstractArray)
    LinearAlgebra.ldiv!(y, s.factorization, x)
    return 0
end
function ldiv!(s::DirectSolver, y::AbstractArray)
    LinearAlgebra.ldiv!(s.factorization, y)
    return 0
end

function batch_ldiv!(s::DirectSolver{<:LinearAlgebra.Factorization}, Y, Js::Vector{SparseMatrixCSC{Float64, Int}}, X)
    nbatch = length(Js)
    for i in 1:nbatch
        lu!(s.factorization, Js[i])
        y = view(Y, :, i)
        x = view(X, :, i)
        LinearAlgebra.ldiv!(y, s.factorization, x)
    end
end

get_transpose(::DirectSolver, M::CUSPARSE.CuSparseMatrixCSR) = CUSPARSE.CuSparseMatrixCSC(M)

function update_preconditioner!(solver::AbstractIterativeLinearSolver, J, device)
    update(solver.precond, J, device)
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
    P = solver.precond.P
    y[:], n_iters, status = bicgstab(J, x, P, y; maxiter=solver.maxiter,
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
    P = solver.precond.P

    y[:], n_iters, status = bicgstab_eigen(J, x, P, y; maxiter=solver.maxiter,
                                           verbose=solver.verbose, tol=solver.tol)
    if status != Converged
        error("EigenBICGSTAB failed to converge. Final status is $(status)")
    end

    return n_iters
end

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
    P = solver.precond.P
    CUDA.allowscalar() do
        Krylov.dqgmres!(solver.inner, J, x; N=P)
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
        Krylov.bicgstab!(solver.inner, J, x;
                                        N=solver.precond.P,
                                        atol=solver.atol,
                                        rtol=solver.rtol,
                                        verbose=solver.verbose)
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
