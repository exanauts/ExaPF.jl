module Iterative

using CUDA
using KernelAbstractions
using IterativeSolvers
using Krylov
using LinearAlgebra
using SparseArrays
using TimerOutputs

using ..ExaPF: Precondition
import ..ExaPF: norm2

export bicgstab, list_solvers

@enum(SolveStatus,
    Unsolved,
    MaxIterations,
    NotANumber,
    Converged
)

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
    xi   = similar(b)
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
    @timeit to "While loop" begin
        while go
            @timeit to "First stage" begin
                rhoi1 = dot(br0, ri) ; beta = (rhoi1/rhoi) * (alpha / omegai)
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
                    @show iter
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

list_solvers(::CPU) = ["bicgstab_ref", "bicgstab", "gmres", "dqgmres", "default"]

function ldiv!(
    dx::AbstractVector,
    J::SparseArrays.SparseMatrixCSC,
    F::AbstractVector,
    solver::String,
    preconditioner=nothing,
    timer::TimerOutputs.TimerOutput=nothing,
)
    if preconditioner != nothing && solver != "default"
        @timeit timer "Preconditioner" P = Precondition.update(J, preconditioner, timer)
        if solver == "bicgstab_ref"
            @timeit timer "CPU-BICGSTAB" (dx[:], history) = IterativeSolvers.bicgstabl(P*J, P*F, log=true)
            n_iters = history.iters
        elseif solver == "bicgstab"
            @timeit timer "GPU-BICGSTAB" dx[:], n_iters = bicgstab(J, F, P, dx, timer, maxiter=10000)
        elseif solver == "gmres"
            @timeit timer "CPU-GMRES" (dx[:], history) = IterativeSolvers.gmres(P*J, P*F, restart=4, log=true)
            n_iters = history.iters
        elseif solver == "dqgmres"
            @timeit timer "GPU-DQGMRES" (dx[:], status) = Krylov.dqgmres(J, F, M=P, memory=4)
            n_iters = length(status.residuals)
        else
            error("Unknown linear solver")
        end
    else
        @timeit timer "CPU-Default sparse solver" dx .= J\F
        n_iters = 0
    end
    return n_iters
end

list_solvers(::CUDADevice) = ["bicgstab", "dqgmres", "default"]

function ldiv!(
    dx::CuVector,
    J::CUDA.CUSPARSE.CuSparseMatrixCSR,
    F::CuVector,
    solver::String,
    preconditioner,
    timer=nothing,
)
    if solver == "bicgstab"
        @timeit timer "Preconditioner" P = Precondition.update(J, preconditioner, timer)
        @timeit timer "GPU-BICGSTAB" dx[:], n_iters, status = bicgstab(J, F, P, dx, timer, maxiter=10000)
        if status != Converged
            error("BICGSTAB failed to converge")
        end
    elseif solver == "dqgmres"
        @timeit timer "Preconditioner" P = Precondition.update(J, preconditioner, timer)
        @timeit timer "GPU-DQGMRES" (dx[:], status) = Krylov.dqgmres(J, F, M=P, memory=4)
        n_iters = length(status.residuals)
    else
        lintol = 1e-8
        @timeit timer "Sparse CUSOLVER" dx  = CUSOLVER.csrlsvqr!(J,F,dx,lintol,one(Cint),'O')
        n_iters = 0
    end
    return n_iters
end

# TODO: pass this function to multiple dispatch
function init_preconditioner(J, solver, npartitions, device; verbose_level=0)
    if solver == "default"
        return Precondition.NoPreconditioner()
    end

    nblock = div(size(J,1), npartitions)
    if verbose_level >= 2
        println("#partitions: $npartitions, Blocksize: n = ", nblock,
                " Mbytes = ", (nblock*nblock*npartitions*8.0)/(1024.0*1024.0))
    end
    precond = Precondition.Preconditioner(J, npartitions, device)
    if verbose_level >= 2
        println("Block Jacobi block size: $(precond.nJs)")
        println("$npartitions partitions created")
    end
    return precond
end

end
