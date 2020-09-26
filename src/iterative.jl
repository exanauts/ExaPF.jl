module Iterative

using CUDA
using KernelAbstractions
using IterativeSolvers
using Krylov
using LinearAlgebra
using SparseArrays
using TimerOutputs

using ..ExaPF: Precondition

export bicgstab, list_solvers

cuzeros = CUDA.zeros

"""
    bicgstab

The bicgstab implementation used for GPU according to 
Van der Vorst, Henk A.
"Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution
of nonsymmetric linear systems."
SIAM Journal on scientific and Statistical Computing 13, no. 2 (1992): 631-644.
"""
function bicgstab(A, b, P, xi, to::TimerOutput;
                  tol=1e-6, maxiter=size(A, 1), verbose=false)
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
        v0 = zeros(Float64, n)
        p0 = zeros(Float64, n)
    else
        v0 = cuzeros(Float64, n)
        p0 = cuzeros(Float64, n)
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
                    verbose && println("Tolerance reached at iteration $iter")
                end

                if maxiter == iter
                    @show iter
                    go = false
                    verbose && println("Not converged")
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

"""
    list_solver(::CPU)

Return the list of linear solvers for the CPU

"""

list_solvers(::CPU) = ["bicgstab_ref", "bicgstab", "gmres", "dqgmres", "default"]

"""
    ldiv!(dx, J, F, solver, preconditioner, timer)

* `dx::AbstractVector`: Solution
* `J::SparseArrays.SparseMatrixCSC`: Input matrix 
* `F::AbstractVector`: RHS
* `solver::String`: A CPU solver 
* `preconditioner=nothing`:
* `timer::TimerOutputs.TimerOutput=nothing`:

Solve the linear system `J * dx = F` 

A valid CPU solver is:

* `bicgstab`: Package internal implementation for GPUs
* `bicgstab_ref`: BiCGSTAB implementation from `IterativeSolvers.jl`
* `gmres`: GMRES implementation from `IterativeSolvers.jl`
* `dqgmres`: DQGMRES implementation from `Krylov.jl`
* `default`: `J\\F`

"""
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

"""
    list_solver(::CPU)

Return the list of linear solvers for the CPU

"""
list_solvers(::CUDADevice) = ["bicgstab", "dqgmres", "default"]

"""
    ldiv!(dx, J, F, solver, preconditioner, timer)

* `dx::AbstractVector`: Solution
* `J::CUDA.CUSPARSE.CuSparseMatrixCSR`: Input matrix on the GPU
* `F::AbstractVector`: RHS
* `solver::String`: A CPU solver 
* `preconditioner=nothing`:
* `timer::TimerOutputs.TimerOutput=nothing`:

Solve the linear system `J * dx = F` 

A valid CPU solver is:

* `bicgstab`: Package internal implementation for GPUs
* `dqgmres`: DQGMRES implementation from `Krylov.jl`
* `default`: `CUSOLVER.csrlsvqr`

"""
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

    nblock = size(J,1) / npartitions
    if verbose_level >= 2
        println("Blocks: $npartitions, Blocksize: n = ", nblock,
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
