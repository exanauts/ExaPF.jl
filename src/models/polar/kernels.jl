# Implement kernels for polar formulation

"""
    power_balance(V, Ybus, Sbus, pv, pq)

Assembly residual function for N-R power flow.
In complex form, the balance equations writes

``
g(V) = V * (Y_{bus} * V)^* - S_{bus}
``

# Note
Code adapted from MATPOWER.
"""
function power_balance(V, Ybus, Sbus, pv, pq)
    # form mismatch vector
    mis = V .* conj(Ybus * V) - Sbus
    # form residual vector
    F = [real(mis[pv])
         real(mis[pq])
         imag(mis[pq]) ]
    return F
end

"""
    residual_jacobian(V, Ybus, pv, pq)

Compute the Jacobian w.r.t. the state `x` of the power
balance function [`power_balance`](@ref).

# Note
Code adapted from MATPOWER.
"""
function residual_jacobian(V, Ybus, ref, pv, pq)
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

function _state_jacobian(polar::PolarForm)
    pf = polar.network
    ref = polar.network.ref
    pv = polar.network.pv
    pq = polar.network.pq
    n = PS.get(pf, PS.NumberOfBuses())

    Y = pf.Ybus
    # Randomized inputs
    Vre = rand(n)
    Vim = rand(n)
    V = Vre .+ 1im .* Vim
    return residual_jacobian(V, Y, ref, pv, pq)
end
_sparsity_pattern(polar::PolarForm) = findnz(_state_jacobian(polar))

"""
    get_power_injection(fr, v_m, v_a, ybus_re, ybus_im)

Computes the power injection at node `fr`.
In polar form, the power injection at node `i` satisfies
```math
p_{i} = \\sum_{j} v_{i} v_{j} (g_{ij} \\cos(\\theta_i - \\theta_j) + b_{ij} \\sin(\\theta_i - \\theta_j))
```
"""
function get_power_injection(fr, v_m, v_a, ybus_re, ybus_im)
    P = 0.0
    for c in ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1
        to = ybus_re.rowval[c]
        aij = v_a[fr] - v_a[to]
        P += v_m[fr]*v_m[to]*(ybus_re.nzval[c]*cos(aij) + ybus_im.nzval[c]*sin(aij))
    end
    return P
end

"""
    get_react_injection(fr, v_m, v_a, ybus_re, ybus_im)

Computes the reactive power injection at node `fr`.
In polar form, the power injection at node `i` satisfies
```math
q_{i} = \\sum_{j} v_{i} v_{j} (g_{ij} \\sin(\\theta_i - \\theta_j) - b_{ij} \\cos(\\theta_i - \\theta_j))
```
"""
function get_react_injection(fr::Int, v_m, v_a, ybus_re::Spmat{VI,VT}, ybus_im::Spmat{VI,VT}) where {VT <: AbstractVector, VI<:AbstractVector}
    Q = zero(eltype(v_m))
    for c in ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1
        to = ybus_re.rowval[c]
        aij = v_a[fr] - v_a[to]
        Q += v_m[fr]*v_m[to]*(ybus_re.nzval[c]*sin(aij) - ybus_im.nzval[c]*cos(aij))
    end
    return Q
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

function residual_polar!(F, v_m, v_a,
                         ybus_re, ybus_im,
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

@kernel function transfer_kernel!(
    vmag, vang, pinj, qinj, u, pv, pq, ref, pload, qload
)
    i = @index(Global, Linear)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    # PV bus
    if i <= npv
        bus = pv[i]
        # P = Pg - Pd
        pinj[bus] = u[nref + i] - pload[bus]
        vmag[bus] = u[nref + npv + i]
    # REF bus
    else
        i_ref = i - npv
        bus = ref[i_ref]
        vmag[bus] = u[i_ref]
        vang[bus] = 0.0  # reference angle set to 0 by default
    end
end

# Transfer values in (x, u) to buffer
function transfer!(polar::PolarForm, buffer::PolarNetworkState, u)
    if isa(u, CuArray)
        kernel! = transfer_kernel!(CUDADevice(), 256)
    else
        kernel! = transfer_kernel!(CPU(), 1)
    end
    nbus = length(buffer.vmag)
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())
    ev = kernel!(
        buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj,
        u,
        pv, pq, ref,
        polar.active_load, polar.reactive_load,
        ndrange=(length(pv)+length(ref)),
    )
    wait(ev)
end

@kernel function active_power_kernel!(
    pg, vmag, vang, pinj, qinj,
    pv, ref, pv_to_gen, ref_to_gen,
    ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
    ybus_im_nzval, pload
)
    i = @index(Global, Linear)
    npv = length(pv)
    nref = length(ref)
    # Evaluate active power at PV nodes
    if i <= npv
        bus = pv[i]
        i_gen = pv_to_gen[i]
        pg[i_gen] = pinj[bus] + pload[bus]
    # Evaluate active power at slack nodes
    elseif i <= npv + nref
        i_ = i - npv
        bus = ref[i_]
        i_gen = ref_to_gen[i_]
        inj = 0
        @inbounds for c in ybus_re_colptr[bus]:ybus_re_colptr[bus+1]-1
            to = ybus_re_rowval[c]
            aij = vang[bus] - vang[to]
            # f_re = a * cos + b * sin
            # f_im = a * sin - b * cos
            coef_cos = vmag[bus]*vmag[to]*ybus_re_nzval[c]
            coef_sin = vmag[bus]*vmag[to]*ybus_im_nzval[c]
            cos_val = cos(aij)
            sin_val = sin(aij)
            inj += coef_cos * cos_val + coef_sin * sin_val
        end
        pg[i_gen] = inj + pload[bus]
    end
end

# Refresh active power (needed to evaluate objective)
function update!(polar::PolarForm, ::PS.Generator, ::PS.ActivePower, buffer::PolarNetworkState)
    if isa(buffer.vmag, Array)
        kernel! = active_power_kernel!(CPU(), 1)
    else
        kernel! = active_power_kernel!(CUDADevice(), 256)
    end
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    range_ = length(pv) + length(ref)

    ev = kernel!(
        buffer.pg,
        buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj,
        pv, ref, pv_to_gen, ref_to_gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
        ybus_im.nzval, polar.active_load,
        ndrange=range_
    )
    wait(ev)
end

@kernel function reactive_power_kernel!(
    qg, vmag, vang, pinj, qinj,
    pv, ref, pv_to_gen, ref_to_gen,
    ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
    ybus_im_nzval, qload
)
    i = @index(Global, Linear)
    npv = length(pv)
    nref = length(ref)
    # Evaluate reactive power at PV nodes
    if i <= npv
        bus = pv[i]
        i_gen = pv_to_gen[i]
    # Evaluate reactive power at slack nodes
    elseif i <= npv + nref
        i_ = i - npv
        bus = ref[i_]
        i_gen = ref_to_gen[i_]
    end
    inj = 0
    @inbounds for c in ybus_re_colptr[bus]:ybus_re_colptr[bus+1]-1
        to = ybus_re_rowval[c]
        aij = vang[bus] - vang[to]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = vmag[bus]*vmag[to]*ybus_re_nzval[c]
        coef_sin = vmag[bus]*vmag[to]*ybus_im_nzval[c]
        cos_val = cos(aij)
        sin_val = sin(aij)
        inj += coef_cos * sin_val - coef_sin * cos_val
    end
    qg[i_gen] = inj + qload[bus]
end

function update!(polar::PolarForm, ::PS.Generator, ::PS.ReactivePower, buffer::PolarNetworkState)
    if isa(buffer.vmag, Array)
        kernel! = reactive_power_kernel!(CPU(), 1)
    else
        kernel! = reactive_power_kernel!(CUDADevice(), 256)
    end
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    range_ = length(pv) + length(ref)
    ev = kernel!(
        buffer.qg,
        buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj,
        pv, ref, pv_to_gen, ref_to_gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
        ybus_im.nzval, polar.reactive_load,
        ndrange=range_
    )
    wait(ev)
end

@kernel function load_power_constraint_kernel!(
    g, qg, ref_to_gen, pv_to_gen, nref, npv, shift
)
    i = @index(Global, Linear)
    # Evaluate reactive power at PV nodes
    if i <= npv
        ig = pv_to_gen[i]
        g[i + nref + shift] = qg[ig]
    else i <= npv + nref
        i_ = i - npv
        ig = ref_to_gen[i_]
        g[i_ + shift] = qg[ig]
    end
end

