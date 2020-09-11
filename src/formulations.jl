
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

    j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
    j12 = real(dSbus_dVm[[pv; pq], pq])
    j21 = imag(dSbus_dVa[pq, [pv; pq]])
    j22 = imag(dSbus_dVm[pq, pq])

    J = [j11 j12; j21 j22]
end

function cost_function(
    pf::PowerSystem.PowerNetwork,
    x::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    device=CPU();
    V=Float64
)
    @warn("Function `ExaPF.cost_function is deprecated")
    # indexes
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
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

    ybus_re, ybus_im = Spmat{T{Int}, T{Float64}}(pf.Ybus)

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

function cost_gradients(
    pf::PowerSystem.PowerNetwork,
    x::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    device=CPU()
)
    @warn("Function `ExaPF.cost_gradients is deprecated")

    # indexes
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
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

    ybus_re, ybus_im = Spmat{T{Int}, T{Float64}}(pf.Ybus)

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

function project_constraints!(u::AbstractArray, grad::AbstractArray, u_min::AbstractArray,
                              u_max::AbstractArray)
    dim = length(u)
    for i=1:dim
        if u[i] > u_max[i]
            @printf("Projecting u[%d] = %f to u_max = %f.\n", i, u[i], u_max[i])
            u[i] = u_max[i]
            grad[i] = 0.0
        elseif u[i] < u_min[i]
            @printf("Projecting u[%d] = %f to u_min = %f.\n", i, u[i], u_min[i])
            u[i] = u_min[i]
            grad[i] = 0.0
        end
    end
end
