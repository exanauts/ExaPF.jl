module LinearSolvers

using CUDA
using KernelAbstractions
using IterativeSolvers
using Krylov
using LinearAlgebra
using Printf
using SparseArrays

import ..ExaPF: xnorm, csclsvqr!
import Base: show

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

get_transpose(::AbstractLinearSolver, M::AbstractMatrix) = transpose(M)

"""
    ldiv!(solver, y, J, x)

* `solver::AbstractLinearSolver`: linear solver to solve the system
* `y::AbstractVector`: Solution
* `J::AbstractMatrix`: Input matrix
* `x::AbstractVector`: RHS

Solve the linear system `J * y = Fx

"""
function ldiv! end

struct DirectSolver <: AbstractLinearSolver end
DirectSolver(precond) = DirectSolver()
function ldiv!(::DirectSolver,
    y::Vector, J::AbstractMatrix, x::Vector,
)
    y .= J \ x
    return 0
end
function ldiv!(::DirectSolver,
    y::CuVector, J::CUDA.CUSPARSE.CuSparseMatrixCSR, x::CuVector,
)
    CUSOLVER.csrlsvqr!(J, x, y, 1e-8, one(Cint), 'O')
    return 0
end
function ldiv!(::DirectSolver,
    y::CuVector, J::CUDA.CUSPARSE.CuSparseMatrixCSC, x::CuVector,
)
    csclsvqr!(J, x, y, 1e-8, one(Cint), 'O')
    return 0
end
get_transpose(::DirectSolver, M::CUDA.CUSPARSE.CuSparseMatrixCSR) = CuSparseMatrixCSC(M)

function update!(solver::AbstractIterativeLinearSolver, J)
    update(J, solver.precond)
end

struct BICGSTAB <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    maxiter::Int
    tol::Float64
    verbose::Bool
end
BICGSTAB(precond; maxiter=2_000, tol=1e-8, verbose=false) = BICGSTAB(precond, maxiter, tol, verbose)
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

struct EigenBICGSTAB <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    maxiter::Int
    tol::Float64
    verbose::Bool
end
EigenBICGSTAB(precond; maxiter=2_000, tol=1e-8, verbose=false) = EigenBICGSTAB(precond, maxiter, tol, verbose)
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

struct RefBICGSTAB <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    verbose::Bool
end
RefBICGSTAB(precond; verbose=true) = RefBICGSTAB(precond, verbose)
function ldiv!(solver::RefBICGSTAB,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    P = solver.precond.P
    y[:], history = IterativeSolvers.bicgstabl(P*J, P*x, log=solver.verbose)
    return history.iters
end

struct RefGMRES <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    restart::Int
    verbose::Bool
end
RefGMRES(precond; restart=4, verbose=true) = RefGMRES(precond, restart, verbose)
function ldiv!(solver::RefGMRES,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    P = solver.precond.P
    y[:], history = IterativeSolvers.gmres(P*J, P*x, restart=solver.restart, log=solver.verbose)
    return history.iters
end

struct DQGMRES <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    memory::Int
    verbose::Bool
end
DQGMRES(precond; memory=4, verbose=false) = DQGMRES(precond, memory, verbose)
function ldiv!(solver::DQGMRES,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    P = solver.precond.P
    (y[:], status) = Krylov.dqgmres(J, x, N=P, memory=solver.memory)
    return length(status.residuals)
end

struct KrylovBICGSTAB <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    verbose::Bool
end
KrylovBICGSTAB(precond; verbose=false) = KrylovBICGSTAB(precond, verbose)
function ldiv!(solver::KrylovBICGSTAB,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    P = solver.precond.P
    (y[:], status) = Krylov.bicgstab(J, x, N=P, atol=1e-10, verbose=false)
    return length(status.residuals)
end


list_solvers(::CPU) = [RefBICGSTAB, RefGMRES, DQGMRES, BICGSTAB, EigenBICGSTAB, DirectSolver]
list_solvers(::CUDADevice) = [BICGSTAB, DQGMRES, EigenBICGSTAB, DirectSolver, KrylovBICGSTAB]

end
