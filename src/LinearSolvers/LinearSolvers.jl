module LinearSolvers

using CUDA
using KernelAbstractions
using IterativeSolvers
using Krylov
using LinearAlgebra
using Printf
using SparseArrays
using TimerOutputs

import ..ExaPF: norm2, TIMER, csclsvqr!
import Base: show

export bicgstab, list_solvers
export DirectSolver, BICGSTAB, EigenBICGSTAB

@enum(SolveStatus,
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
    y::Vector, J::AbstractSparseMatrix, x::Vector,
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

function update!(solver::AbstractIterativeLinearSolver, J)
    @timeit solver.timer "Preconditioner"  update(J, solver.precond, solver.timer)
end

struct BICGSTAB <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    maxiter::Int
    tol::Float64
    verbose::Bool
    timer::TimerOutput
end
BICGSTAB(precond; maxiter=2_000, tol=1e-8, verbose=false) = BICGSTAB(precond, maxiter, tol, verbose, TIMER)
function ldiv!(solver::BICGSTAB,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    P = solver.precond.P

    @timeit solver.timer "BICGSTAB" begin
        y[:], n_iters, status = bicgstab(J, x, P, y, solver.timer; maxiter=solver.maxiter,
                                         verbose=solver.verbose, tol=solver.tol)
    end
    if status != Converged
        error("BICGSTAB failed to converge. Final status is $(status)")
    end
    return n_iters
end

struct EigenBICGSTAB <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    maxiter::Int
    tol::Float64
    verbose::Bool
    timer::TimerOutput
end
EigenBICGSTAB(precond; maxiter=10_000, tol=1e-8, verbose=false) = EigenBICGSTAB(precond, maxiter, tol, verbose, TIMER)
function ldiv!(solver::EigenBICGSTAB,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    P = solver.precond.P

    @timeit solver.timer "BICGSTAB" begin
        y[:], n_iters, status = bicgstab_eigen(J, x, P, y, solver.timer; maxiter=solver.maxiter,
                                               verbose=solver.verbose, tol=solver.tol)
    end
    if status != Converged
        error("EigenBICGSTAB failed to converge. Final status is $(status)")
    end

    return n_iters
end

struct RefBICGSTAB <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    verbose::Bool
    timer::TimerOutput
end
RefBICGSTAB(precond; verbose=true) = RefBICGSTAB(precond, verbose, TIMER)
function ldiv!(solver::RefBICGSTAB,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    P = solver.precond.P
    @timeit solver.timer "CPU-BICGSTAB" begin
        y[:], history = IterativeSolvers.bicgstabl(P*J, P*x, log=solver.verbose)
    end
    return history.iters
end

struct RefGMRES <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    restart::Int
    verbose::Bool
    timer::TimerOutput
end
RefGMRES(precond; restart=4, verbose=true) = RefGMRES(precond, restart, verbose, TIMER)
function ldiv!(solver::RefGMRES,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    P = solver.precond.P
    @timeit solver.timer "CPU-GMRES" begin
        y[:], history = IterativeSolvers.gmres(P*J, P*x, restart=solver.restart, log=solver.verbose)
    end
    return history.iters
end

struct DQGMRES <: AbstractIterativeLinearSolver
    precond::AbstractPreconditioner
    memory::Int
    verbose::Bool
    timer::TimerOutput
end
DQGMRES(precond; memory=4, verbose=false) = DQGMRES(precond, memory, verbose, TIMER)
function ldiv!(solver::DQGMRES,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    P = solver.precond.P
    @timeit solver.timer "GPU-DQGMRES" begin
        (y[:], status) = Krylov.dqgmres(J, x, M=P, memory=solver.memory)
    end
    return length(status.residuals)
end


list_solvers(::CPU) = [RefBICGSTAB, RefGMRES, DQGMRES, BICGSTAB, EigenBICGSTAB, DirectSolver]
list_solvers(::CUDADevice) = [BICGSTAB, DQGMRES, EigenBICGSTAB, DirectSolver]

end
