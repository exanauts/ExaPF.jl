import KernelAbstractions: @index
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

KA.@kernel function residual_kernel!(F, v_m, v_a,
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
        kernel! = residual_kernel!(KA.CPU(), 4)
    else
        kernel! = residual_kernel!(KA.CUDADevice(), 256)
    end
    ev = kernel!(F, v_m, v_a,
                 ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
                 ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
                 pinj, qinj, pv, pq, nbus,
                 ndrange=npv+npq)
    wait(ev)
end

KA.@kernel function residual_adj_kernel!(F, adj_F, v_m, adj_v_m, v_a, adj_v_a,
                                  ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                                  ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                                  pinj, adj_pinj, qinj, adj_qinj, pv, pq, nbus)

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
        # Forward loop
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
        if i > npv
            F[npq + i] += coef_cos * sin_val - coef_sin * cos_val
        end
        # Reverse loop
        adj_coef_cos = 0.0
        adj_coef_sin = 0.0
        adj_cos_val  = 0.0
        adj_sin_val  = 0.0

        if i > npv
            adj_coef_cos +=  sin_val  * adj_F[npq + i]
            adj_coef_sin += -cos_val  * adj_F[npq + i]
            adj_cos_val  += -coef_sin * adj_F[npq + i]
            adj_sin_val  +=  coef_cos * adj_F[npq + i]
        end

        adj_coef_cos +=  cos_val  * adj_F[i]
        adj_coef_sin +=  sin_val  * adj_F[i]
        adj_cos_val  +=  coef_cos * adj_F[i]
        adj_sin_val  +=  coef_sin * adj_F[i]

        adj_aij =   cos(aij)*adj_sin_val
        adj_aij += -sin(aij)*adj_cos_val

        adj_v_m[fr] += v_m[to]*ybus_im_nzval[c]*adj_coef_sin
        adj_v_m[to] += v_m[fr]*ybus_im_nzval[c]*adj_coef_sin
        adj_v_m[fr] += v_m[to]*ybus_re_nzval[c]*adj_coef_cos
        adj_v_m[to] += v_m[fr]*ybus_re_nzval[c]*adj_coef_cos

        adj_v_a[fr] += adj_aij
        adj_v_a[to] -= adj_aij
    end
    if i > npv
        adj_qinj[fr] -= adj_F[i + npq]
    end
    adj_pinj[fr] -= adj_F[i]
end

function residual_adj_polar!(F, adj_F, v_m, adj_v_m, v_a, adj_v_a,
                         ybus_re, ybus_im,
                         pinj, adj_pinj, qinj, adj_qinj, pv, pq, nbus)
    npv = length(pv)
    npq = length(pq)
    if isa(F, Array)
        kernel! = residual_adj_kernel!(KA.CPU(), 4)
    else
        kernel! = residual_adj_kernel!(KA.CUDADevice(), 256)
    end
    ev = kernel!(F, adj_F, v_m, adj_v_m, v_a, adj_v_a,
                 ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
                 ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
                 pinj, adj_pinj, qinj, adj_qinj, pv, pq, nbus,
                 ndrange=npv+npq)
    wait(ev)
end

KA.@kernel function transfer_kernel!(
    vmag, vang, pinj, qinj, u, pv, pq, ref, pload, qload
)
    i = @index(Global, Linear)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    # PV bus
    if i <= npv
        bus = pv[i]
        vmag[bus] = u[nref + i]
        # P = Pg - Pd
        pinj[bus] = u[nref + npv + i] - pload[bus]
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
    if isa(u, CUDA.CuArray)
        kernel! = transfer_kernel!(KA.CUDADevice(), 256)
    else
        kernel! = transfer_kernel!(KA.CPU(), 1)
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

KA.@kernel function active_power_kernel!(
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
        kernel! = active_power_kernel!(KA.CPU(), 1)
    else
        kernel! = active_power_kernel!(KA.CUDADevice(), 256)
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

KA.@kernel function reactive_power_kernel!(
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
        kernel! = reactive_power_kernel!(KA.CPU(), 1)
    else
        kernel! = reactive_power_kernel!(KA.CUDADevice(), 256)
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

KA.@kernel function load_power_constraint_kernel!(
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

KA.@kernel function branch_flow_kernel!(
        slines, vmag, vang,
        yff_re, yft_re, ytf_re, ytt_re,
        yff_im, yft_im, ytf_im, ytt_im,
        f, t, nlines,
   )
    ℓ = @index(Global, Linear)
    fr_bus = f[ℓ]
    to_bus = t[ℓ]

    Δθ = vang[fr_bus] - vang[to_bus]
    cosθ = cos(Δθ)
    sinθ = sin(Δθ)

    # branch apparent power limits - from bus
    yff_abs = yff_re[ℓ]^2 + yff_im[ℓ]^2
    yft_abs = yft_re[ℓ]^2 + yft_im[ℓ]^2
    yre_fr =   yff_re[ℓ] * yft_re[ℓ] + yff_im[ℓ] * yft_im[ℓ]
    yim_fr = - yff_re[ℓ] * yft_im[ℓ] + yff_im[ℓ] * yft_re[ℓ]

    fr_flow = vmag[fr_bus]^2 * (
        yff_abs * vmag[fr_bus]^2 + yft_abs * vmag[to_bus]^2 +
        2 * vmag[fr_bus] * vmag[to_bus] * (yre_fr * cosθ - yim_fr * sinθ)
    )
    slines[ℓ] = fr_flow

    # branch apparent power limits - to bus
    ytf_abs = ytf_re[ℓ]^2 + ytf_im[ℓ]^2
    ytt_abs = ytt_re[ℓ]^2 + ytt_im[ℓ]^2
    yre_to =   ytf_re[ℓ] * ytt_re[ℓ] + ytf_im[ℓ] * ytt_im[ℓ]
    yim_to = - ytf_re[ℓ] * ytt_im[ℓ] + ytf_im[ℓ] * ytt_re[ℓ]

    to_flow = vmag[to_bus]^2 * (
        ytf_abs * vmag[fr_bus]^2 + ytt_abs * vmag[to_bus]^2 +
        2 * vmag[fr_bus] * vmag[to_bus] * (yre_to * cosθ - yim_to * sinθ)
    )
    slines[ℓ + nlines] = to_flow
end

function branch_flow_kernel_zygote(
        yff_re, yft_re, ytf_re, ytt_re,
        yff_im, yft_im, ytf_im, ytt_im,
        fr_vmag, to_vmag,
        cosθ, sinθ
   )

    # branch apparent power limits - from bus
    yff_abs = yff_re.^2 .+ yff_im.^2
    yft_abs = yft_re.^2 .+ yft_im.^2
    yre_fr =   yff_re .* yft_re .+ yff_im .* yft_im
    yim_fr = .- yff_re .* yft_im .+ yff_im .* yft_re

    fr_flow = fr_vmag.^2 .* (
        yff_abs .* fr_vmag.^2 .+ yft_abs .* to_vmag.^2 .+
        2 .* fr_vmag .* to_vmag .* (yre_fr .* cosθ .- yim_fr .* sinθ)
    )

    # branch apparent power limits - to bus
    ytf_abs = ytf_re.^2 + ytf_im.^2
    ytt_abs = ytt_re.^2 + ytt_im.^2
    yre_to =   ytf_re .* ytt_re .+ ytf_im .* ytt_im
    yim_to = - ytf_re .* ytt_im .+ ytf_im .* ytt_re

    to_flow = to_vmag.^2 .* (
        ytf_abs .* fr_vmag.^2 .+ ytt_abs .* to_vmag.^2 .+
        2 .* fr_vmag .* to_vmag .* (yre_to .* cosθ .- yim_to .* sinθ)
    )
    return vcat(fr_flow, to_flow)
end

"""
    accumulate_view(x, vx, indices)

.+= is broken on views with redundant indices in CUDA.jl leading to a bug Zygote (see #89).
This implements the .+= operator.

"""
KA.@kernel function accumulate_view!(x, vx, indices)
    # This is parallelizable
    id = @index(Global, Linear)
    for (j, idx) in enumerate(indices)
        if id == idx
            x[id] += vx[j]
        end
    end
end

