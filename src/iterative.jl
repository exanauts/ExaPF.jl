module Iterative

using CUDA
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using TimerOutputs

include("algorithms/precondition.jl")
using .Precondition

export bicgstab

cuzeros = CUDA.zeros

"""
bicgstab according to

Van der Vorst, Henk A.
"Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems."
SIAM Journal on scientific and Statistical Computing 13, no. 2 (1992): 631-644.
"""
function bicgstab(A, b, P, xi, to = nothing; tol = 1e-6, maxiter = size(A,1),
                  verbose=false)
    # parameters
    n    = size(b, 1)
    x0   = similar(b)
    x0 = P * b
    r0   = b - A * x0
    br0  = copy(r0)
    rho0 = 1.0
    alpha = 1.0
    omega0 = 1.0
    if A isa SparseArrays.SparseMatrixCSC
        v0   = p0 = zeros(Float64, n)
    else
        v0   = p0 = cuzeros(Float64, n)
    end

    ri     = copy(r0)
    rhoi   = copy(rho0)
    omegai = copy(omega0)
    vi = copy(v0)
    pi = copy(p0)
    xi .= x0

    y = similar(pi)
    s = similar(pi)
    z = similar(pi)
    t1 = similar(pi)
    t2 = similar(pi)
    pi1 = similar(pi)
    vi1 = similar(pi)
    xi1 = similar(pi)
    t = similar(pi)

    go = true
    iter = 1
    @timeit to "While loop" begin
        while go
            @timeit to "First stage" begin
                rhoi1 = dot(br0, ri) ; beta = (rhoi1/rhoi) * (alpha / omegai)
                pi1 .= ri .+ beta .* (pi .- omegai .* vi)
                y .= P * pi1
                vi1 .= A * y
                alpha = rhoi1 / dot(br0, vi1)
                s .= ri .- (alpha * vi1)
                z .= P * s
                t .= A * z
                t1 .= P * t
                t2 .= P * s
            end
            @timeit to "Main stage" begin
                omegai1 = dot(t1, t2) / dot(t1, t1)
                xi1 .= xi .+ alpha .* y .+ omegai1 .* z
            end

            @timeit to "End stage" begin
                anorm = norm((A * xi1) .- b)

                if verbose
                    println("\tIteration: ", iter)
                    println("\tAbsolute norm: ", anorm)
                end

                if anorm < tol
                    go = false
                    println("Tolerance reached at iteration $iter")
                end

                if maxiter == iter
                    @show iter
                    go = false
                    println("Not converged")
                end

                ri     .= s .- omegai1 .* t
                rhoi   = rhoi1
                pi     .= pi1
                vi     .= vi1
                omegai = omegai1
                xi     .= xi1
                iter   += 1
            end
        end
    end

    return xi, iter
end

function ldiv!(
    dx::AbstractVector,
    J::SparseArrays.SparseMatrixCSC,
    F::AbstractVector,
    solver::String,
    preconditioner,
    timer::TimerOutputs.TimerOutput=nothing,
)
    if solver == "bicgstab_ref"
        @timeit timer "Preconditioner" P = Precondition.update(J, preconditioner, timer)
        @timeit timer "CPU-BICGSTAB" (dx[:], history) = IterativeSolvers.bicgstabl(P*J, P*F, log=true)
        n_iters = history.iters
    elseif solver == "bicgstab"
        @timeit timer "Preconditioner" P = Precondition.update(J, preconditioner, timer)
        @timeit timer "GPU-BICGSTAB" dx[:], n_iters = bicgstab(J, F, P, dx, timer, maxiter=10000)
    elseif solver == "gmres"
        @timeit timer "Preconditioner" P = Precondition.update(J, preconditioner, timer)
        @timeit timer "CPU-GMRES" (dx[:], history) = IterativeSolvers.gmres(P*J, P*F, log=true)
        n_iters = history.iters
    else
        @timeit timer "CPU-Default sparse solver" dx .= J\F
        n_iters = 0
    end
    return n_iters
end

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
        @timeit timer "GPU-BICGSTAB" dx[:], n_iters = bicgstab(J, F, P, dx, timer, maxiter=10000)
    else
        lintol = 1e-8
        @timeit timer "Sparse CUSOLVER" dx  = CUSOLVER.csrlsvqr!(J,F,dx,lintol,one(Cint),'O')
        n_iters = 0
    end
    return n_iters
end

end
