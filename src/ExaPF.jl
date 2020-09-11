# Power flow module. The implementation is a modification of
# MATPOWER's code. We attach the original MATPOWER's license in makeYbus.m:
#
# MATPOWER
# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC)
# by Ray Zimmerman, PSERC Cornell
#
# Covered by the 3-clause BSD License.
module ExaPF

using CUDA
using CUDA.CUSPARSE
using CUDA.CUSOLVER
using ForwardDiff
using IterativeSolvers
using KernelAbstractions
using Krylov
using LinearAlgebra
using Printf
using SparseArrays
using SparseDiffTools
using TimerOutputs

export solve

include("utils.jl")
# Import submodules
include("ad.jl")
using .AD
include("algorithms/precondition.jl")
using .Precondition
include("indexes.jl")
using .IndexSet
include("iterative.jl")
using .Iterative
include("parsers/parse_mat.jl")
using .ParseMAT
include("parsers/parse_psse.jl")
using .ParsePSSE
include("PowerSystem/PowerSystem.jl")
using .PowerSystem

const PS = PowerSystem

include("formulations.jl")
# Modeling
include("models/models.jl")
include("evaluators.jl")

const TIMER = TimerOutput()

const VERBOSE_LEVEL_HIGH = 3
const VERBOSE_LEVEL_MEDIUM = 2
const VERBOSE_LEVEL_LOW = 1
const VERBOSE_LEVEL_NONE = 0


function solve(
    pf::PowerSystem.PowerNetwork,
    x::AbstractArray,
    u::AbstractArray,
    p::AbstractArray;
    npartitions=2,
    solver="default",
    tol=1e-7,
    maxiter=20,
    device=CPU(),
    verbose_level=0,
)
    @warn("Function `ExaPF.solve is deprecated. Use ExaPF.powerflow instead.")
    # Set array type
    # For CPU choose Vector and SparseMatrixCSC
    # For GPU choose CuVector and SparseMatrixCSR (CSR!!! Not CSC)
    if verbose_level >= VERBOSE_LEVEL_LOW
        println("Target set to device $(device)")
    end
    if isa(device, CPU)
        T = Vector
        M = SparseMatrixCSC
        A = Array
    elseif isa(device, CUDADevice)
        T = CuVector
        M = CuSparseMatrixCSR
        A = CuArray
    else
        error("Only `CPU` and `CUDADevice` are supported.")
    end

    # Retrieve parameter and initial voltage guess
    data = pf.data
    Ybus = pf.Ybus
    vmag, vang, pinj, qinj = PowerSystem.retrieve_physics(pf, x, u, p)

    # Convert vectors to target
    V = T(pf.vbus)
    Vm = T(vmag)
    Va = T(vang)
    pbus = T(pinj)
    qbus = T(qinj)

    # iteration variables
    iter = 0
    converged = false

    ybus_re, ybus_im = Spmat{T{Int}, T{Float64}}(Ybus)

    nbus = pf.nbus
    ngen = pf.ngen

    ref = pf.ref
    pv = pf.pv
    pq = pf.pq

    # retrieve ref, pv and pq index
    pv = T(pv)
    pq = T(pq)

    # indices
    npv = size(pv, 1);
    npq = size(pq, 1);
    j1 = 1
    j2 = npq
    j3 = j2 + 1
    j4 = j2 + npq
    j5 = j4 + 1
    j6 = j4 + npv

    # form residual function
    F = T(zeros(Float64, npv + 2*npq))
    dx = similar(F)

    # Evaluate residual function
    residualFunction_polar!(F, Vm, Va,
                            ybus_re, ybus_im,
                            pbus, qbus, pv, pq, nbus)
    # Build the AD Jacobian structure
    stateJacobianAD = AD.StateJacobianAD(F, Vm, Va,
                                         ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus)
    designJacobianAD = AD.DesignJacobianAD(F, Vm, Va,
                                           ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus)
    if verbose_level >= VERBOSE_LEVEL_MEDIUM
        print("State Jacobian  --- ")
        println(stateJacobianAD)
        print("Design Jacobian --- ")
        println(designJacobianAD)
    end

    J = stateJacobianAD.J
    preconditioner = Precondition.NoPreconditioner()
    if solver != "default"
        nblock = size(J,1) / npartitions
        if verbose_level >= VERBOSE_LEVEL_MEDIUM
            println("Blocks: $npartitions, Blocksize: n = ", nblock,
                    " Mbytes = ", (nblock*nblock*npartitions*8.0)/1024.0/1024.0)
            println("Partitioning...")
        end
        preconditioner = Precondition.Preconditioner(J, npartitions, device)
        if verbose_level >= VERBOSE_LEVEL_MEDIUM
            println("$npartitions partitions created")
        end
    end

    # check for convergence
    normF = norm(F, Inf)
    if verbose_level >= VERBOSE_LEVEL_HIGH
        @printf("Iteration %d. Residual norm: %g.\n", iter, normF)
    end
    if normF < tol
        converged = true
    end

    linsol_iters = Int[]
    dx = T{Float64}(undef, size(J,1))
    Vapv = view(Va, pv)
    Vapq = view(Va, pq)
    Vmpq = view(Vm, pq)
    dx12 = view(dx, j1:j2)
    dx34 = view(dx, j3:j4)
    dx56 = view(dx, j5:j6)

    @timeit TIMER "Newton" while ((!converged) && (iter < maxiter))

        iter += 1

        @timeit TIMER "Jacobian" begin
            AD.residualJacobianAD!(stateJacobianAD, residualFunction_polar!, Vm, Va,
                                   ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus, TIMER)
        end
        J = stateJacobianAD.J
        # J = residualJacobian(V, Ybus, pv, pq)

        # Find descent direction
        n_iters = Iterative.ldiv!(dx, J, F, solver, preconditioner, TIMER)
        push!(linsol_iters, n_iters)
        # Sometimes it is better to move backward
        dx .= -dx

        # update voltage
        @timeit TIMER "Update voltage" begin
            if (npv != 0)
                # Va[pv] .= Va[pv] .+ dx[j5:j6]
                Vapv .= Vapv .+ dx56
            end
            if (npq != 0)
                # Vm[pq] .= Vm[pq] .+ dx[j1:j2]
                Vmpq .= Vmpq .+ dx12
                # Va[pq] .= Va[pq] .+ dx[j3:j4]
                Vapq .= Vapq .+ dx34
            end
        end

        @timeit TIMER "Exponential" V .= Vm .* exp.(1im .* Va)

        @timeit TIMER "Angle and magnitude" begin
            polar!(Vm, Va, V, device)
        end

        F .= 0.0
        @timeit TIMER "Residual function" begin
            residualFunction_polar!(F, Vm, Va,
                ybus_re, ybus_im,
                pbus, qbus, pv, pq, nbus)
        end

        @timeit TIMER "Norm" normF = norm(F, Inf)
        if verbose_level >= VERBOSE_LEVEL_HIGH
            @printf("Iteration %d. Residual norm: %g.\n", iter, normF)
        end

        if normF < tol
            converged = true
        end
    end

    if verbose_level >= VERBOSE_LEVEL_HIGH
        if converged
            @printf("N-R converged in %d iterations.\n", iter)
        else
            @printf("N-R did not converge.\n")
        end
    end

    xk = PowerSystem.get_x(pf, Vm, Va, pbus, qbus)

    # Timer outputs display
    if verbose_level >= VERBOSE_LEVEL_MEDIUM
        show(TIMER)
        println("")
    end
    reset_timer!(TIMER)
    conv = ConvergenceStatus(converged, iter, normF, sum(linsol_iters))
    # Build closures
    function Ju(pf, x, u, p)
        Vm, Va, pbus, qbus = PowerSystem.retrieve_physics(pf, x, u, p)
        AD.residualJacobianAD!(designJacobianAD, residualFunction_polar!, Vm, Va,
                                ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus, TIMER)
        return designJacobianAD.J
    end
    function Jx(pf, x, u, p)
        Vm, Va, pbus, qbus = PowerSystem.retrieve_physics(pf, x, u, p)
        AD.residualJacobianAD!(stateJacobianAD, residualFunction_polar!, Vm, Va,
                                ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus, TIMER)
        return stateJacobianAD.J
    end
    function g(pf, x, u, p)
        Vm, Va, pbus, qbus = PowerSystem.retrieve_physics(pf, x, u, p)
        residualFunction_polar!(F, Vm, Va,
                                ybus_re,
                                ybus_im,
                                pbus, qbus, pv, pq, nbus)
        return F
    end
    function residualFunction_x!(vecx)
        x_ = Vector{eltype(vecx)}(undef, length(x))
        u_ = Vector{eltype(vecx)}(undef, length(u))
        x_ .= vecx[1:length(x)]
        u_ .= vecx[length(x)+1:end]
        F_ = Vector{eltype(vecx)}(undef, length(F))
        F_ .= 0
        Vm, Va, pbus, qbus = ExaPF.PowerSystem.retrieve_physics(pf, x_, u_, p; V=eltype(vecx))
        residualFunction_polar!(F_, Vm, Va,
                                ybus_re,
                                ybus_im,
                                pbus, qbus, pv, pq, nbus)
        return F_
    end
    return xk, g, Jx, Ju, conv, residualFunction_x!
end
end
