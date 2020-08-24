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
include("helpers.jl")
include("ad.jl")
using .AD
include("algorithms/precondition.jl")
using .Precondition
include("indexes.jl")
using .IndexSet
include("iterative.jl")
using .Iterative
include("parse/parse_mat.jl")
using .ParseMAT
include("parse/parse_psse.jl")
using .ParsePSSE
include("powersystem.jl")
using .PowerSystem

const TIMER = TimerOutput()


struct ConvergenceStatus
    has_converged::Bool
    n_iterations::Int
    norm_residuals::Float64
    n_linear_solves::Int
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
function residualFunction_polar_sparsity!(F, v_m, v_a,
                     ybus_re, ybus_im,
                     pinj, qinj, pv, pq, nbus)

    npv = size(pv, 1)
    npq = size(pq, 1)

    for i in 1:length(pv)+length(pq)
        # REAL PQ: (npv+1:npv+npq)
        # IMAG PQ: (npv+npq+1:npv+2npq)
        fr = (i <= npv) ? pv[i] : pq[i - npv]
        F[i] -= pinj[fr]
        if i > npv
            F[i + npq] -= qinj[fr]
        end
        for c in ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1
            to = ybus_re.rowval[c]
            aij = v_a[fr] - v_a[to]
            # f_re = a * cos + b * sin
            # f_im = a * sin - b * cos
            coef_cos = v_m[fr]*v_m[to]*ybus_re.nzval[c]
            coef_sin = v_m[fr]*v_m[to]*ybus_im.nzval[c]
            cos_val = cos(aij)
            sin_val = sin(aij)
            F[i] += coef_cos * cos_val + coef_sin * sin_val
            if i > npv
                F[npq + i] += coef_cos * sin_val - coef_sin * cos_val
            end
        end
    end
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
                     ybus_re,
                     ybus_im,
                     pinj, qinj, pv, pq, nbus)
    npv = length(pv)
    npq = length(pq)
    if isa(F, Array)
        kernel! = residual_kernel!(CPU(), 4)
    else
        kernel! = residual_kernel!(CUDADevice(), 256)
    end
    ev = kernel!(F, v_m, v_a,
                 ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
                 ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
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

    j11 = real(dSbus_dVm[[pv; pq], pq])
    j12 = real(dSbus_dVa[[pv; pq], [pq; pv]])
    j21 = imag(dSbus_dVm[pq, pq])
    j22 = imag(dSbus_dVa[pq, [pq; pv]])

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

function project_constraints!(u::AbstractArray, grad::AbstractArray, u_min::AbstractArray,
                              u_max::AbstractArray)
    dim = length(u)
    for i=1:dim
        if u[i] > u_max[i]
            @printf("Projecting u[%d] = %f to u_max = %f.\n", i, u[i], u_max[i])
            u[i] = u_max[i]
            grad[i] = 0.0
        elseif u[i] < u_min[i]
            @printf("Projecting u[%d] = %f to u_max = %f.\n", i, u[i], u_max[i])
            u[i] = u_max[i]
            grad[i] = 0.0
        end
    end
end

"""
    get_constraints(pf)

Given PowerNetwork object, returns vectors xmin, xmax, umin, umax of the OPF box constraints.

"""
function get_constraints(pf::PowerSystem.PowerNetwork)

    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()
    GEN_BUS, PG, QG, QMAX, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()

    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)
    b2i = pf.bus_to_indexes

    gens = pf.data["gen"]
    baseMVA = pf.data["baseMVA"][1]
    bus = pf.data["bus"]
    ngens = size(gens)[1]

    dimension_u = 2*npv + nref
    dimension_x = 2*npq + npv

    u_min = fill(-Inf, dimension_u)
    u_max = fill(Inf, dimension_u)
    x_min = fill(-Inf, dimension_x)
    x_max = fill(Inf, dimension_x)
    p_min = fill(-Inf, nref)
    p_max = fill(Inf, nref)

    for i in 1:length(pf.pq)
        bus_idx = pf.pq[i]
        vm_max = bus[bus_idx, VMAX]
        vm_min = bus[bus_idx, VMIN]
        x_min[i] = vm_min
        x_max[i] = vm_max
    end

    for i in 1:length(pf.pv)
        bus_idx = pf.pv[i]
        vm_max = bus[bus_idx, VMAX]
        vm_min = bus[bus_idx, VMIN]
        u_min[nref + npv + i] = vm_min
        u_max[nref + npv + i] = vm_max
    end

    for i in 1:length(pf.ref)
        bus_idx = pf.ref[i]
        vm_max = bus[bus_idx, VMAX]
        vm_min = bus[bus_idx, VMIN]
        u_min[i] = vm_min
        u_max[i] = vm_max
    end

    for i = 1:ngens
        genbus = b2i[gens[i, GEN_BUS]]
        bustype = bus[genbus, BUS_TYPE]
        if bustype == PowerSystem.PV_BUS_TYPE
            idx_pv = findfirst(pf.pv.==genbus)
            u_min[nref + idx_pv] = gens[i, PMIN] / baseMVA
            u_max[nref + idx_pv] = gens[i, PMAX] / baseMVA
        elseif bustype == PowerSystem.REF_BUS_TYPE
            idx = findfirst(pf.ref .== genbus)
            p_min[idx] = gens[i, PMIN] / baseMVA
            p_max[idx] = gens[i, PMAX] / baseMVA
        end
    end

    return u_min, u_max, x_min, x_max, p_min, p_max
end

function cost_function(pf::PowerSystem.PowerNetwork, x::AbstractArray, u::AbstractArray,
              p::AbstractArray, device=CPU(); V=Float64)

    # indexes
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()
    GEN_BUS, PG, QG, QMAX, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()
    MODEL, STARTUP, SHUTDOWN, NCOST, COST = IndexSet.idx_cost()

    # Set array type
    # For CPU choose Vector and SparseMatrixCSC
    # For GPU choose CuVector and SparseMatrixCSR (CSR!!! Not CSC)
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
    vmag, vang, pinj, qinj = ExaPF.PowerSystem.retrieve_physics(pf, x, u, p; V=V)

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
    nbus = size(bus)[1]

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
            pinj_ref = PowerSystem.get_power_injection(genbus, vmag, vang, ybus_re, ybus_im)
            cost += c0 + c1*pinj_ref*baseMVA + c2*(pinj_ref*baseMVA)^2
        end

    end

    # Dommel an Tinney recommend to increase S every iteration
	s = 0.0
	for i = 1:nbus
		bustype = bus[i, BUS_TYPE]
		if bustype == 1
			vm_max = bus[i, VMAX]
			vm_min = bus[i, VMIN]
			if vmag[i] > vm_max
				cost += s*(vmag[i] - vm_max)^2
			elseif vmag[i] < vm_min
                cost += s*(vm_min - vmag[i])^2
			end
		end
	end

    return cost
end

function cost_gradients(pf::PowerSystem.PowerNetwork, x::AbstractArray, u::AbstractArray,
              p::AbstractArray, device=CPU())

    # indexes
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()
    GEN_BUS, PG, QG, QMAX, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()
    MODEL, STARTUP, SHUTDOWN, NCOST, COST = IndexSet.idx_cost()

    # Set array type
    # For CPU choose Vector and SparseMatrixCSC
    # For GPU choose CuVector and SparseMatrixCSR (CSR!!! Not CSC)
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

    nref = length(ref)
    npv = length(pv)
    npq = length(pq)

    ybus_re, ybus_im = Spmat{T}(pf.Ybus)

    # matpower assumes gens are ordered. Genrator in row i has its cost on row i
    # of the cost table.
    gens = pf.data["gen"]
    baseMVA = pf.data["baseMVA"][1]
    bus = pf.data["bus"]
    cost_data = pf.data["cost"]
    ngens = size(gens)[1]


    dCdx = zeros(length(x))
    dCdu = zeros(length(u))

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
            # This is shameful. Cannot think of a better way to do it r.n
            idx_pv = findall(pv.==genbus)[1]
            dCdu[nref + idx_pv] = c1*baseMVA + 2*c2*(baseMVA^2)*pinj[genbus]
        elseif bustype == 3
            # let c_i(x, u) = c0 + c1*baseMVA*f(x, u) + c2*(baseMVA*f(x, u))^2
            # c_i'(x, u) = c1*baseMVA*f'(x, u) + 2*c2*baseMVA*f(x, u)*f'(x, u)
            idx_ref = findall(ref.==genbus)[1]
            dPdVm, dPdVa = PowerSystem.get_power_injection_partials(genbus, vmag, vang, ybus_re, ybus_im)
            pinj_ref = PowerSystem.get_power_injection(genbus, vmag, vang, ybus_re, ybus_im)

            dCdx[1:npq] += c1*baseMVA*dPdVm[pq] + 2*c2*(baseMVA^2)*pinj_ref*dPdVm[pq]
            dCdx[npq + 1:2*npq] += c1*baseMVA*dPdVa[pq] + 2*c2*(baseMVA^2)*pinj_ref*dPdVa[pq]
            dCdx[2*npq + 1:2*npq + npv] += c1*baseMVA*dPdVa[pv] + 2*c2*(baseMVA^2)*pinj_ref*dPdVa[pv]

            dCdu[1:nref] += c1*baseMVA*dPdVm[ref] + 2*c2*(baseMVA^2)*pinj_ref*dPdVm[ref]
            dCdu[nref + npv + 1:nref + 2*npv] += (c1*baseMVA*dPdVm[pv]
                                                  + 2*c2*(baseMVA^2)*pinj_ref*dPdVm[pv])
        end

    end
    return dCdx, dCdu
end

function solve(pf::PowerSystem.PowerNetwork,
    x::AbstractArray,
    u::AbstractArray,
    p::AbstractArray;
    npartitions=2,
    solver="default",
    tol=1e-7,
    maxiter=20,
    device=CPU(),
    verbose=false
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

    ybus_re, ybus_im = Spmat{T}(Ybus)

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
    stateJacobianAD = AD.StateJacobianAD(residualFunction_polar_sparsity!, F, Vm, Va,
                                         ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus)
    designJacobianAD = AD.DesignJacobianAD(residualFunction_polar_sparsity!, F, Vm, Va,
                                           ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus)
    J = stateJacobianAD.J
    preconditioner = Precondition.NoPreconditioner()
    if solver != "default"
        nblock = size(J,1) / npartitions
        println("Blocks: $npartitions, Blocksize: n = ", nblock,
                " Mbytes = ", (nblock*nblock*npartitions*8.0)/1024.0/1024.0)
        println("Partitioning...")
        preconditioner = Precondition.Preconditioner(J, npartitions, device)
        println("$npartitions partitions created")
    end

    # check for convergence
    normF = norm(F, Inf)
    @printf("Iteration %d. Residual norm: %g.\n", iter, normF)

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

    xk = PowerSystem.get_x(pf, Vm, Va, pbus, qbus)

    # Timer outputs display
    if verbose
        show(TIMER)
        println("") #this really bugs me
    end
    reset_timer!(TIMER)
    conv = ConvergenceStatus(converged, iter, normF, sum(linsol_iters))
    function Ju(pf, x, u, p)
        Vm, Va, pbus, qbus = PowerSystem.retrieve_physics(pf, x, u, p)
        AD.designJacobianAD!(designJacobianAD, residualFunction_polar!, Vm, Va,
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
