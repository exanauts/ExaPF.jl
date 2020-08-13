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

# Import submodules
include("ad.jl")
using .AD
include("algorithms/precondition.jl")
using .Precondition
include("indexes.jl")
using .IdxSet
include("iterative.jl")
using .Iterative
include("parse/parse_mat.jl")
using .ParseMAT
include("parse/parse_psse.jl")
using .ParsePSSE
include("powersystem.jl")
using .PowerSystem

const TIMER = TimerOutput()


mutable struct Spmat{T}
    colptr
    rowval
    nzval

    # function spmat{T}(colptr::Vector{Int64}, rowval::Vector{Int64}, nzval::Vector{T}) where T
    function Spmat{T}(mat::SparseMatrixCSC{Complex{Float64}, Int}) where T
        matreal = new(T{Int64}(mat.colptr), T{Int64}(mat.rowval), T{Float64}(real.(mat.nzval)))
        matimag = new(T{Int64}(mat.colptr), T{Int64}(mat.rowval), T{Float64}(imag.(mat.nzval)))
        return matreal, matimag
    end
end

"""
residualFunction

Assembly residual function for N-R power flow
"""
function residualFunction(V, Ybus, Sbus, pv, pq)
    # form mismatch vector
    mis = V .* conj(Ybus * V) - Sbus
    # form residual vector
    F = [real(mis[pv])
         real(mis[pq])
         imag(mis[pq]) ]
    return F
end

function residualFunction_real!(F, v_re, v_im,
                                ybus_re, ybus_im, pinj, qinj, pv, pq, nbus)

    npv = size(pv, 1)
    npq = size(pq, 1)

    # REAL PV
    for i in 1:npv
        fr = pv[i]
        F[i] -= pinj[fr]
        for (j,c) in enumerate(ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1)
            to = ybus_re.rowval[c]
            F[i] += (v_re[fr]*(v_re[to]*ybus_re.nzval[c] - v_im[to]*ybus_im.nzval[c]) +
                     v_im[fr]*(v_im[to]*ybus_re.nzval[c] + v_re[to]*ybus_im.nzval[c]))
        end
    end

    # REAL PQ
    for i in 1:npq
        fr = pq[i]
        F[npv + i] -= pinj[fr]
        for (j,c) in enumerate(ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1)
            to = ybus_re.rowval[c]
            F[npv + i] += (v_re[fr]*(v_re[to]*ybus_re.nzval[c] - v_im[to]*ybus_im.nzval[c]) +
                           v_im[fr]*(v_im[to]*ybus_re.nzval[c] + v_re[to]*ybus_im.nzval[c]))
        end
    end

    # IMAG PQ
    for i in 1:npq
        fr = pq[i]
        F[npv + npq + i] -= qinj[fr]
        for (j,c) in enumerate(ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1)
            to = ybus_re.rowval[c]
            F[npv + npq + i] += (v_im[fr]*(v_re[to]*ybus_re.nzval[c] - v_im[to]*ybus_im.nzval[c]) -
                                 v_re[fr]*(v_im[to]*ybus_re.nzval[c] + v_re[to]*ybus_im.nzval[c]))
        end
    end

    return F
end

@kernel function residual_kernel!(F, v_m, v_a,
                     ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                     ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                     pinj, qinj, pv, pq, nbus)

    npv = size(pv, 1)
    npq = size(pq, 1)

    i = @index(Global, Linear)
    # REAL PV: 1:npv
    # REAL PQ: (npv+1:npv+npq)
    # IMAG PQ: (npv+npq+1:npv+2npq)
    fr = (i <= npv) ? pv[i] : pq[i - npv]
    F[i] -= pinj[fr]
    if i > npv
        F[i + npq] -= qinj[fr]
    end
    @inbounds for c in ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1
        to = ybus_re_rowval[c]
        aij = v_a[fr] - v_a[to]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = v_m[fr]*v_m[to]*ybus_re_nzval[c]
        coef_sin = v_m[fr]*v_m[to]*ybus_im_nzval[c]
        cos_val = cos(aij)
        sin_val = sin(aij)
        F[i] += coef_cos * cos_val + coef_sin * sin_val
        if i > npv
            F[npq + i] += coef_cos * sin_val - coef_sin * cos_val
        end
    end
end

function residualFunction_polar!(F, v_m, v_a,
                     ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                     ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                     pinj, qinj, pv, pq, nbus)
    npv = length(pv)
    npq = length(pq)
    if isa(F, Array)
        kernel! = residual_kernel!(CPU(), 4)
    else
        kernel! = residual_kernel!(CUDADevice(), 256)
    end
    ev = kernel!(F, v_m, v_a,
                 ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                 ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                 pinj, qinj, pv, pq, nbus,
                 ndrange=npv+npq)
    wait(ev)
end

function residualJacobian(V, Ybus, pv, pq)
    n = size(V, 1)
    Ibus = Ybus*V
    diagV       = sparse(1:n, 1:n, V, n, n)
    diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
    diagVnorm   = sparse(1:n, 1:n, V./abs.(V), n, n)

    dSbus_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
    dSbus_dVa = 1im * diagV * conj(diagIbus - Ybus * diagV)

    j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
    j12 = real(dSbus_dVm[[pv; pq], pq])
    j21 = imag(dSbus_dVa[pq, [pv; pq]])
    j22 = imag(dSbus_dVm[pq, pq])

    J = [j11 j12; j21 j22]
end

# small utils function
function polar!(Vm, Va, V, ::CPU)
    Vm .= abs.(V)
    Va .= angle.(V)
end
function polar!(Vm, Va, V, ::CUDADevice)
    Vm .= CUDA.abs.(V)
    Va .= CUDA.angle.(V)
end

function get_power_injection(fr, v_m, v_a, ybus_re, ybus_im)

    P = 0.0
    for (j,c) in enumerate(ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1)
        to = ybus_re.rowval[c]
        aij = v_a[fr] - v_a[to]
        P += v_m[fr]*v_m[to]*(ybus_re.nzval[c]*cos(aij) + ybus_im.nzval[c]*sin(aij))
    end

    return P
end

function cost(pf::PowerSystem.PowerNetwork, x::AbstractArray, u::AbstractArray,
              p::AbstractArray, device=CPU())
    
    # indexes
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IdxSet.idx_bus()
    GEN_BUS, PG, QG, QMAX, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IdxSet.idx_gen()
    MODEL, STARTUP, SHUTDOWN, NCOST, COST = IdxSet.idx_cost()
    
    # Set array type
    # For CPU choose Vector and SparseMatrixCSC
    # For GPU choose CuVector and SparseMatrixCSR (CSR!!! Not CSC)
    println("Target set to device $(device)")
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

    # for now, let's just return the sum of all generator power
    vmag, vang, pinj, qinj = ExaPF.PowerSystem.retrieve_physics(pf, x, u, p)
    
    ref = pf.ref
    pv = pf.pv
    pq = pf.pq
    b2i = pf.bus_to_indexes
    
    ybus_re, ybus_im = Spmat{T}(pf.Ybus)
    
    # matpower assumes gens are ordered. Genrator in row i has its cost on row i
    # of the cost table.
    gens = pf.data["gen"]
    baseMVA = pf.data["baseMVA"][1]
    bus = pf.data["bus"]
    cost_data = pf.data["cost"]
    ngens = size(gens)[1]

    # initialize cost
    cost = 0.0
    
    # iterate generators and check if pv or ref.
    for i = 1:ngens
        # only 2nd degree polynomial implemented for now.
        @assert cost_data[i, MODEL] == 2
        @assert cost_data[i, NCOST] == 3
        genbus = b2i[gens[i, GEN_BUS]]
        bustype = bus[genbus, BUS_TYPE]

        # polynomial coefficients
        c0 = cost_data[i, COST][3] 
        c1 = cost_data[i, COST][2] 
        c2 = cost_data[i, COST][1]

        if bustype == 2
            cost += c0 + c1*pinj[genbus]*baseMVA + c2*(pinj[genbus]*baseMVA)^2
        elseif bustype == 3
            pinj_ref = get_power_injection(genbus, vmag, vang, ybus_re, ybus_im)
            cost += c0 + c1*pinj_ref*baseMVA + c2*(pinj_ref*baseMVA)^2
        end
            
    end

    return cost
end

function solve(pf::PowerSystem.PowerNetwork,
    x::AbstractArray,
    u::AbstractArray,
    p::AbstractArray;
    npartitions=2,
    solver="default",
    tol=1e-6,
    maxiter=20,
    device=CPU()
)
    # Set array type
    # For CPU choose Vector and SparseMatrixCSC
    # For GPU choose CuVector and SparseMatrixCSR (CSR!!! Not CSC)
    println("Target set to device $(device)")
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
    V = pf.V
    data = pf.data
    Ybus = pf.Ybus

    # Convert voltage vector to target
    V = T(V)

    # iteration variables
    iter = 0
    converged = false

    ybus_re, ybus_im = Spmat{T}(Ybus)

    nbus = pf.nbus
    ngen = pf.ngen

    ref = pf.ref
    pv = pf.pv
    pq = pf.pq

    # retrieve ref, pv and pq index
    pv = T(pv)
    pq = T(pq)

    # retrieve power injections
    Sbus = pf.Sbus
    pbus = T(real(Sbus))
    qbus = T(imag(Sbus))

    # initiate voltage
    Vm, Va = similar(V, Float64), similar(V, Float64)
    polar!(Vm, Va, V, device)

    # indices
    npv = size(pv, 1);
    npq = size(pq, 1);
    j1 = 1
    j2 = npv
    j3 = j2 + 1
    j4 = j2 + npq
    j5 = j4 + 1
    j6 = j4 + npq

    # form residual function
    F = T(zeros(Float64, npv + 2*npq))
    dx = similar(F)

    # Evaluate residual function
    residualFunction_polar!(F, Vm, Va,
                            ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
                            ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
                            pbus, qbus, pv, pq, nbus)
    # Initiate coloring for AD
    J = residualJacobian(V, Ybus, pv, pq)
    dim_J = size(J, 1)
    preconditioner = Precondition.NoPreconditioner()
    if solver != "default"
        nblock = size(J,1) / npartitions
        println("Blocks: $npartitions, Blocksize: n = ", nblock,
                " Mbytes = ", (nblock*nblock*npartitions*8.0)/1024.0/1024.0)
        println("Partitioning...")
        preconditioner = Precondition.Preconditioner(J, npartitions, device)
        println("$npartitions partitions created")
    end

    println("Coloring...")
    @timeit TIMER "Coloring" coloring = T{Int64}(matrix_colors(J))
    ncolors = size(unique(coloring),1)
    println("Number of Jacobian colors: ", ncolors)
    println("Creating JacobianAD...")
    J = M(J)
    stateJacobianAD = AD.StateJacobianAD(J, coloring, F, Vm, Va, pbus, pv, pq, ref)
    # designJacobianAD = AD.DesignJacobianAD(J, coloring, F, Vm, Va, pbus, pv, pq, ref)

    # check for convergence
    normF = norm(F, Inf)
    @printf("Iteration %d. Residual norm: %g.\n", iter, normF)

    if normF < tol
        converged = true
    end

    linsol_iters = []
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
        J = residualJacobian(V, Ybus, pv, pq)

        # Find descent direction
        n_iters = Iterative.ldiv!(dx, J, F, solver, preconditioner, TIMER)
        push!(linsol_iters, n_iters)
        # Sometimes it is better to move backward
        dx .= -dx

        # update voltage
        @timeit TIMER "Update voltage" begin
            if (npv != 0)
                # Va[pv] .= Va[pv] .+ dx[j1:j2]
                Vapv .= Vapv .+ dx12
            end
            if (npq != 0)
                # Va[pq] .= Va[pq] .+ dx[j3:j4]
                Vapq .= Vapq .+ dx34
                # Vm[pq] .= Vm[pq] .+ dx[j5:j6]
                Vmpq .= Vmpq .+ dx56
            end
        end

        @timeit TIMER "Exponential" V .= Vm .* exp.(1im .* Va)

        @timeit TIMER "Angle and magnitude" begin
            polar!(Vm, Va, V, device)
        end

        F .= 0.0
        @timeit TIMER "Residual function" begin
            residualFunction_polar!(F, Vm, Va,
                ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
                ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
                pbus, qbus, pv, pq, nbus)
        end

        @timeit TIMER "Norm" normF = norm(F, Inf)
        @printf("Iteration %d. Residual norm: %g.\n", iter, normF)

        if normF < tol
            converged = true
        end
    end

    if converged
        @printf("N-R converged in %d iterations.\n", iter)
    else
        @printf("N-R did not converge.\n")
    end

    # Timer outputs display
    show(TIMER)
    reset_timer!(TIMER)
    # AD.designJacobianAD!(designJacobianAD, residualFunction_polar!, Vm, Va,
    #                         ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus, TIMER)
    return V, converged, normF, linsol_iters[1], sum(linsol_iters)#, designJacobianAD.J
end

# end of module
end
