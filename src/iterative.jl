module Iterative

using CUDA
using KernelAbstractions
using IterativeSolvers
using Krylov
using LinearAlgebra
using SparseArrays
using TimerOutputs

using ..ExaPF: Precondition
import ..ExaPF: norm2, TIMER

export bicgstab, list_solvers
export DirectSolver, BICGSTAB, EigenBICGSTAB

@enum(SolveStatus,
    Unsolved,
    MaxIterations,
    NotANumber,
    Converged
)

abstract type AbstractLinearSolver end
abstract type AbstractIterativeLinearSolver <: AbstractLinearSolver end

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

function update!(solver::AbstractIterativeLinearSolver, J)
    @timeit solver.timer "Preconditioner"  Precondition.update(J, solver.precond, solver.timer)
end

struct BICGSTAB <: AbstractIterativeLinearSolver
    precond::Precondition.AbstractPreconditioner
    maxiter::Int
    tol::Float64
    verbose::Bool
    timer::TimerOutput
end
BICGSTAB(precond; maxiter=10_000, tol=1e-8, verbose=false) = BICGSTAB(precond, maxiter, tol, verbose, TIMER)
function ldiv!(solver::BICGSTAB,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector,
)
    P = solver.precond.P
    if P isa SparseMatrixCSC
        if any(isnan.(P.nzval))
            error("NaNs in P")
        end
        if any(isnan.(J.nzval))
            error("NaNs in J")
        end
    else
        if any(isnan.(P.nzVal))
            error("NaNs in P")
        end
        if any(isnan.(J.nzVal))
            error("NaNs in J")
        end
    end

    @timeit solver.timer "BICGSTAB" begin
        y[:], n_iters, status = bicgstab(J, x, P, y, solver.timer; maxiter=solver.maxiter,
                                         verbose=solver.verbose, tol=solver.tol)
    end
    if any(isnan.(y))
        error("NaNs in y")
    end
    return n_iters
end

struct EigenBICGSTAB <: AbstractIterativeLinearSolver
    precond::Precondition.AbstractPreconditioner
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
    if P isa SparseMatrixCSC
        if any(isnan.(P.nzval))
            error("NaNs in P")
        end
        if any(isnan.(J.nzval))
            error("NaNs in J")
        end
    else
        if any(isnan.(P.nzVal))
            error("NaNs in P")
        end
        if any(isnan.(J.nzVal))
            error("NaNs in J")
        end
    end

    @timeit solver.timer "BICGSTAB" begin
        y[:], n_iters, status = bicgstab_eigen(J, x, P, y, solver.timer; maxiter=solver.maxiter,
                                         verbose=solver.verbose, tol=solver.tol)
    end

    if any(isnan.(y))
        error("NaNs in y")
    end

    return n_iters
end
struct RefBICGSTAB <: AbstractIterativeLinearSolver
    precond::Precondition.AbstractPreconditioner
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
    precond::Precondition.AbstractPreconditioner
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
    precond::Precondition.AbstractPreconditioner
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

function bicgstab_eigen(A, b, P, x, to::TimerOutput;
                  tol=1e-8, maxiter=size(A, 1), verbose=false)

    mul!(x, P, b)
    r  = b .- A * x
    r0 = similar(r)
    r0 .= r
    
    r0_sqnorm = norm(r0)^2
    rhs_sqnorm = norm(b)^2
    if rhs_sqnorm == 0
        x .= 0.0
        return x, 0.0, Converged
    end
    rho    = 1.0
    alpha  = 1.0
    w      = 1.0
    
    v = similar(x); p = similar(x)
    fill!(v, 0.0); fill!(p, 0.0)

    y = similar(x); z = similar(x)
    s = similar(x); t = similar(x)

    tol2 = tol*tol*rhs_sqnorm
    eps2 = eps(Float64)^2
    i = 0
    restarts = 0
    status = Unsolved

    while norm(r)^2 > tol2 && i < maxiter
        rho_old = rho

        rho = dot(r0, r)

        if abs(rho) < eps2*r0_sqnorm
            mul!(r, A, x)
            r .= b .- r
            r0 .= r
            v .= 0.0
            p .= 0.0
            rho = norm(r)^2
            r0_sqnorm = norm(r)^2
            if restarts == 0
                i = 0
            end
            restarts += 1
        end

        beta = (rho/rho_old) * (alpha / w)
        p .= r .+ (beta * (p .- w .* v))
        

        mul!(y, P, p)
        mul!(v, A, y)

        alpha = rho / dot(r0, v)
        s .= r .- alpha .* v

        mul!(z, P, s)
        mul!(t, A, z)

        tmp = norm(t)^2
        if tmp > 0.0
            w = dot(t,s) / tmp
        else
            w = 0.0
        end
        x .= x .+ alpha * y .+ w * z;
        r .= s .- w * t;
        i += 1
    end
    if maxiter == i
        go = false
        status = MaxIterations
        verbose && println("Restarts: $restarts")
        verbose && println("Not converged")
    end
    if norm(r)^2 <= tol2
        go = false
        status = Converged
        verbose && println("Restarts: $restarts")
        verbose && println("Tolerance reached at iteration $i")
    end
    return x, i, status
end

"""
bicgstab according to

Van der Vorst, Henk A.
"Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution
of nonsymmetric linear systems."
SIAM Journal on scientific and Statistical Computing 13, no. 2 (1992): 631-644.
"""
function bicgstab(A, b, P, xi, to::TimerOutput;
                  tol=1e-8, maxiter=size(A, 1), verbose=false)
    # parameters
    n    = size(b, 1)
    mul!(xi, P, b)
    ri   = b - A * xi
    br0  = copy(ri)
    rho0 = 1.0
    alpha = 1.0
    omega0 = 1.0
    vi = similar(xi)
    pi = similar(xi)
    fill!(vi, 0)
    fill!(pi, 0)

    rhoi   = copy(rho0)
    omegai = copy(omega0)
    residual = copy(b)

    y = similar(pi)
    s = similar(pi)
    z = similar(pi)
    t1 = similar(pi)
    t2 = similar(pi)
    pi1 = similar(pi)
    vi1 = similar(pi)
    t = similar(pi)

    go = true
    status = Unsolved
    iter = 1
    restarts = 0
    @timeit to "While loop" begin
        while go
            @timeit to "First stage" begin
                rhoi1 = dot(br0, ri) ; 
                if abs(rhoi1) < 1e-20
                    restarts += 1
                    ri .= b - A * xi
                    br0 .= ri
                    residual .= b
                    rho0 = 1.0
                    rhoi = rho0
                    rhoi1 = dot(br0,ri)
                    alpha = 1.0
                    omega0 = 1.0
                    omegai = 1.0
                    fill!(vi, 0.0)
                    fill!(pi, 0.0)
                end
                beta = (rhoi1/rhoi) * (alpha / omegai)
                pi1 .= ri .+ beta .* (pi .- omegai .* vi)
                mul!(y, P, pi1)
                mul!(vi1, A, y)
                alpha = rhoi1 / dot(br0, vi1)
                s .= ri .- (alpha .* vi1)

                mul!(z, P, s)
                mul!(t, A, z)
                mul!(t1, P, t)
                mul!(t2, P, s)
            end
            @timeit to "Main stage" begin
                omegai1 = dot(t1, t2) / dot(t1, t1)
                xi .= xi .+ alpha .* y .+ omegai1 .* z
            end

            @timeit to "End stage" begin
                # TODO: should update to five arguments mul!
                #       once CUDA.jl 1.4 is released
                # mul!(residual, A, xi, 1.0, -1.0)
                residual .= A * xi .- b
                anorm = norm2(residual)

                if verbose
                    println("\tIteration: ", iter)
                    println("\tAbsolute norm: ", anorm)
                end

                if isnan(anorm)
                    go = false
                    status = NotANumber
                end
                if anorm < tol
                    go = false
                    status = Converged
                    verbose && println("Tolerance reached at iteration $iter")
                end
                if maxiter == iter
                    go = false
                    status = MaxIterations
                    verbose && println("Not converged")
                end

                ri     .= s .- omegai1 .* t
                rhoi   = rhoi1
                pi     .= pi1
                vi     .= vi1
                omegai = omegai1
                iter   += 1
            end
        end
    end

    return xi, iter, status
end

list_solvers(::CPU) = [RefBICGSTAB, RefGMRES, DQGMRES, BICGSTAB, EigenBICGSTAB, DirectSolver]
list_solvers(::CUDADevice) = [BICGSTAB, DQGMRES, EigenBICGSTAB, DirectSolver]

end
