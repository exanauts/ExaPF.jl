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

KA.@kernel function residual_kernel!(F, vm, va,
                                  colptr, rowval,
                                  ybus_re_nzval, ybus_im_nzval,
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
    @inbounds for c in colptr[fr]:colptr[fr+1]-1
        to = rowval[c]
        aij = va[fr] - va[to]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = vm[fr]*vm[to]*ybus_re_nzval[c]
        coef_sin = vm[fr]*vm[to]*ybus_im_nzval[c]
        cos_val = cos(aij)
        sin_val = sin(aij)
        F[i] += coef_cos * cos_val + coef_sin * sin_val
        if i > npv
            F[npq + i] += coef_cos * sin_val - coef_sin * cos_val
        end
    end
end

function residual_polar!(F, vm, va, pinj, qinj,
                         ybus_re, ybus_im,
                         pv, pq, ref, nbus)
    npv = length(pv)
    npq = length(pq)
    if isa(F, Array)
        kernel! = residual_kernel!(KA.CPU())
    else
        kernel! = residual_kernel!(KA.CUDADevice())
    end
    ev = kernel!(F, vm, va,
                 ybus_re.colptr, ybus_re.rowval,
                 ybus_re.nzval, ybus_im.nzval,
                 pinj, qinj, pv, pq, nbus,
                 ndrange=npv+npq)
    wait(ev)
end

KA.@kernel function adj_residual_edge_kernel!(
    F, adj_F, vm, adj_vm, va, adj_va,
    colptr, rowval,
    ybus_re_nzval, ybus_im_nzval,
    edge_vm_from, edge_vm_to,
    edge_va_from, edge_va_to,
    pinj, adj_pinj, qinj, pv, pq
)

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
    @inbounds for c in colptr[fr]:colptr[fr+1]-1
        # Forward loop
        to = rowval[c]
        aij = va[fr] - va[to]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = vm[fr]*vm[to]*ybus_re_nzval[c]
        coef_sin = vm[fr]*vm[to]*ybus_im_nzval[c]

        cos_val = cos(aij)
        sin_val = sin(aij)
        F[i] += coef_cos * cos_val + coef_sin * sin_val
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

        adj_aij =   cos_val*adj_sin_val
        adj_aij += -sin_val*adj_cos_val

        edge_vm_from[c] += vm[to]*ybus_im_nzval[c]*adj_coef_sin
        edge_vm_to[c]   += vm[fr]*ybus_im_nzval[c]*adj_coef_sin
        edge_vm_from[c] += vm[to]*ybus_re_nzval[c]*adj_coef_cos
        edge_vm_to[c]   += vm[fr]*ybus_re_nzval[c]*adj_coef_cos

        edge_va_from[c] += adj_aij
        edge_va_to[c]   -= adj_aij
    end
    # qinj is not active
    # if i > npv
    #     adj_qinj[fr] -= adj_F[i + npq]
    # end
    adj_pinj[fr] -= adj_F[i]
end

KA.@kernel function adj_residual_node_kernel!(F, adj_F, vm, adj_vm, va, adj_va,
                                  colptr, rowval,
                                  ybus_re_nzval, ybus_im_nzval,
                                  edge_vm_from, edge_vm_to,
                                  edge_va_from, edge_va_to,
                                  pinj, adj_pinj, qinj, pv, pq)

    npv = size(pv, 1)
    npq = size(pq, 1)

    i = @index(Global, Linear)
    @inbounds for c in colptr[i]:colptr[i+1]-1
        adj_vm[i] += edge_vm_from[c]
        adj_vm[i] += edge_vm_to[c]
        adj_va[i] += edge_va_from[c]
        adj_va[i] += edge_va_to[c]
    end
end

function adj_residual_polar!(
    F, adj_F, vm, adj_vm, va, adj_va,
    ybus_re, ybus_im,
    pinj, adj_pinj, qinj, pv, pq, nbus
)
    npv = length(pv)
    npq = length(pq)
    nvbus = length(vm)
    nnz = length(ybus_re.nzval)
    T = eltype(vm)
    edge_vm_from = zeros(T, nnz)
    edge_vm_to   = zeros(T, nnz)
    edge_va_from = zeros(T, nnz)
    edge_va_to   = zeros(T, nnz)
    colptr = ybus_re.colptr
    rowval = ybus_re.rowval
    if isa(F, Array)
        kernel_edge! = adj_residual_edge_kernel!(KA.CPU())
        kernel_node! = adj_residual_node_kernel!(KA.CPU())
        spedge_vm_to = SparseMatrixCSC{T, Int64}(nvbus, nvbus, colptr, rowval, edge_vm_to)
        spedge_va_to = SparseMatrixCSC{T, Int64}(nvbus, nvbus, colptr, rowval, edge_va_to)
    else
        kernel_edge! = adj_residual_edge_kernel!(KA.CUDADevice())
        kernel_node! = adj_residual_node_kernel!(KA.CUDADevice())
    end
    ev = kernel_edge!(F, adj_F, vm, adj_vm, va, adj_va,
                 ybus_re.colptr, ybus_re.rowval,
                 ybus_re.nzval, ybus_im.nzval,
                 edge_vm_from, edge_vm_to,
                 edge_va_from, edge_va_to,
                 pinj, adj_pinj, qinj, pv, pq,
                 ndrange=npv+npq)
    wait(ev)

    spedge_vm_to.nzval .= edge_vm_to
    spedge_va_to.nzval .= edge_va_to

    transpose!(spedge_vm_to, copy(spedge_vm_to))
    transpose!(spedge_va_to, copy(spedge_va_to))
    ev = kernel_node!(F, adj_F, vm, adj_vm, va, adj_va,
                 ybus_re.colptr, ybus_re.rowval,
                 ybus_re.nzval, ybus_im.nzval,
                 edge_vm_from, spedge_vm_to.nzval,
                 edge_va_from, spedge_va_to.nzval,
                 pinj, adj_pinj, qinj, pv, pq,
                 ndrange=nvbus)
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
        kernel! = transfer_kernel!(KA.CUDADevice())
    else
        kernel! = transfer_kernel!(KA.CPU())
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

KA.@kernel function adj_transfer_kernel!(
    adj_u, adj_x, adj_vmag, adj_vang, adj_pinj, adj_qinj, pv, pq, ref,
)
    i = @index(Global, Linear)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    # PQ buses
    if i <= npq
        bus = pq[i]
        adj_x[npv+i] =  adj_vang[bus]
        adj_x[npv+npq+i] = adj_vmag[bus]
    # PV buses
    elseif i <= npq + npv
        i_ = i - npq
        bus = pv[i_]
        adj_u[nref + i_] = adj_vmag[bus]
        adj_u[nref + npv + i_] += adj_pinj[bus]
        adj_x[i_] = adj_vang[bus]
    # SLACK buses
    elseif i <= npq + npv + nref
        i_ = i - npq - npv
        bus = ref[i_]
        adj_u[i_] = adj_vmag[bus]
    end
end

# Transfer values in (x, u) to buffer
function adjoint_transfer!(
    polar::PolarForm,
    ∂u, ∂x,
    ∂vm, ∂va, ∂pinj, ∂qinj,
)
    nbus = get(polar, PS.NumberOfBuses())
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    ev = adj_transfer_kernel!(polar.device)(
        ∂u, ∂x,
        ∂vm, ∂va, ∂pinj, ∂qinj,
        pv, pq, ref;
        ndrange=nbus,
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
function update!(polar::PolarForm, ::PS.Generators, ::PS.ActivePower, buffer::PolarNetworkState)
    if isa(buffer.vmag, Array)
        kernel! = active_power_kernel!(KA.CPU())
    else
        kernel! = active_power_kernel!(KA.CUDADevice())
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

KA.@kernel function adj_active_power_edge_kernel!(
    pg, adj_pg,
    vmag, adj_vmag, vang, adj_vang,
    pinj, adj_pinj, qinj, adj_qinj,
    pv, ref, pv_to_gen, ref_to_gen,
    edge_vmag_bus, edge_vmag_to,
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
    if i <= npv
        bus = pv[i]
        i_gen = pv_to_gen[i]
        adj_pinj[bus] = adj_pg[i_gen]
        adj_pload[bus] = adj_pg[i_gen]
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

        adj_inj = adj_pg[i_gen]
        adj_pload[bus] = adj_pg[i_gen]
        @inbounds for c in ybus_re_colptr[bus]:ybus_re_colptr[bus+1]-1
            to = ybus_re_rowval[c]
            aij = vang[bus] - vang[to]
            # f_re = a * cos + b * sin
            # f_im = a * sin - b * cos
            coef_cos = vmag[bus]*vmag[to]*ybus_re_nzval[c]
            coef_sin = vmag[bus]*vmag[to]*ybus_im_nzval[c]
            cos_val = cos(aij)
            sin_val = sin(aij)

            adj_coef_cos = cos_val  * adj_inj
            adj_cos_val  = coef_cos * adj_inj
            adj_coef_sin = sin_val  * adj_inj
            adj_sin_val  = coef_sin * adj_inj

            adj_aij =   cos_val*adj_sin_val
            adj_aij += -sin_val*adj_cos_val

            edge_vmag_bus[c] += vmag[to] *ybus_re_nzval[c]*adj_coef_cos
            edge_vmag_to[c]  += vmag[bus]*ybus_re_nzval[c]*adj_coef_cos
            edge_vmag_bus[c] += vmag[to] *ybus_im_nzval[c]*adj_coef_sin
            edge_vmag_to[c]  += vmag[bus]*ybus_im_nzval[c]*adj_coef_sin
        end
    end
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

function update!(polar::PolarForm, ::PS.Generators, ::PS.ReactivePower, buffer::PolarNetworkState)
    if isa(buffer.vmag, Array)
        kernel! = reactive_power_kernel!(KA.CPU())
    else
        kernel! = reactive_power_kernel!(KA.CUDADevice())
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

KA.@kernel function adj_reactive_power_edge_kernel!(
    qg, adj_qg,
    vmag, adj_vmag, vang, adj_vang,
    pinj, adj_pinj, qinj, adj_qinj,
    pv, ref, pv_to_gen, ref_to_gen,
    edge_vmag_bus, edge_vmag_to,
    edge_vang_bus, edge_vang_to,
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
    inj = 0.0
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

    # Reverse run
    adj_inj = adj_qg[i_gen]
    @inbounds for c in ybus_re_colptr[bus]:ybus_re_colptr[bus+1]-1
        to = ybus_re_rowval[c]
        aij = vang[bus] - vang[to]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = vmag[bus]*vmag[to]*ybus_re_nzval[c]
        coef_sin = vmag[bus]*vmag[to]*ybus_im_nzval[c]
        cos_val = cos(aij)
        sin_val = sin(aij)

        adj_coef_cos = sin_val  * adj_inj
        adj_sin_val  = coef_cos * adj_inj
        adj_coef_sin = -cos_val  * adj_inj
        adj_cos_val  = -coef_sin * adj_inj

        adj_aij =   coef_cos * cos_val * adj_inj
        adj_aij +=  coef_sin * sin_val * adj_inj

        edge_vmag_bus[c] += vmag[to] *ybus_re_nzval[c]*adj_coef_cos
        edge_vmag_to[c]  += vmag[bus]*ybus_re_nzval[c]*adj_coef_cos
        edge_vmag_bus[c] += vmag[to] *ybus_im_nzval[c]*adj_coef_sin
        edge_vmag_to[c]  += vmag[bus]*ybus_im_nzval[c]*adj_coef_sin

        edge_vang_bus[c] += adj_aij
        edge_vang_to[c]  -= adj_aij
    end
end


KA.@kernel function adj_reactive_power_node_kernel!(
    pg, adj_pg,
    vmag, adj_vmag, vang, adj_vang,
    pinj, adj_pinj, qinj, adj_qinj,
    pv, ref, pv_to_gen, ref_to_gen,
    edge_vmag_bus, edge_vmag_to,
    edge_vang_bus, edge_vang_to,
    colptr, rowval,
)

    npv = size(pv, 1)

    i = @index(Global, Linear)
    @inbounds for c in colptr[i]:colptr[i+1]-1
        to = rowval[c]
        adj_vmag[i] += edge_vmag_bus[c]
        adj_vmag[i] += edge_vmag_to[c]
        adj_vang[i] += edge_vang_bus[c]
        adj_vang[i] += edge_vang_to[c]
    end
end


# Refresh active power (needed to evaluate objective)
function adj_reactive_power!(
    F, adj_F, vm, adj_vm, va, adj_va,
    ybus_re, ybus_im,
    pinj, adj_pinj, qinj, adj_qinj, reactive_load,
    pv, pq, ref, pv_to_gen, ref_to_gen, nbus
)
    npv = length(pv)
    npq = length(pq)
    nvbus = length(vm)
    nnz = length(ybus_re.nzval)
    T = eltype(adj_vm)

    edge_vmag_bus = zeros(T, nnz)
    edge_vmag_to   = zeros(T, nnz)
    edge_vang_bus = zeros(T, nnz)
    edge_vang_to   = zeros(T, nnz)

    colptr = ybus_re.colptr
    rowval = ybus_re.rowval

    if isa(vm, CUDA.CuArray) # TODO
        kernel_edge! = adj_reactive_power_edge_kernel!(KA.CUDADevice())
        kernel_node! = adj_reactive_power_node_kernel!(KA.CUDADevice())
    else
        kernel_edge! = adj_reactive_power_edge_kernel!(KA.CPU())
        kernel_node! = adj_reactive_power_node_kernel!(KA.CPU())
        spedge_vmag_to = SparseMatrixCSC{T, Int64}(nvbus, nvbus, colptr, rowval, edge_vmag_to)
        spedge_vang_to = SparseMatrixCSC{T, Int64}(nvbus, nvbus, colptr, rowval, edge_vang_to)
    end

    range_ = length(pv) + length(ref)

    ev = kernel_edge!(
        F, adj_F,
        vm, adj_vm,
        va, adj_va,
        pinj, adj_pinj,
        qinj, adj_qinj,
        pv, ref, pv_to_gen, ref_to_gen,
        edge_vmag_bus, edge_vmag_to,
        edge_vang_bus, edge_vang_to,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
        ybus_im.nzval, reactive_load,
        ndrange=range_
    )
    wait(ev)

    spedge_vmag_to.nzval .= edge_vmag_to
    spedge_vang_to.nzval .= edge_vang_to
    transpose!(spedge_vmag_to, copy(spedge_vmag_to))
    transpose!(spedge_vang_to, copy(spedge_vang_to))

    ev = kernel_node!(
        F, adj_F,
        vm, adj_vm,
        va, adj_va,
        pinj, adj_pinj,
        qinj, adj_qinj,
        pv, ref, pv_to_gen, ref_to_gen,
        edge_vmag_bus, spedge_vmag_to.nzval,
        edge_vang_bus, spedge_vang_to.nzval,
        ybus_re.colptr, ybus_re.rowval,
        ndrange=nvbus
    )
    wait(ev)
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
        2.0 * vmag[fr_bus] * vmag[to_bus] * (yre_fr * cosθ - yim_fr * sinθ)
    )
    slines[ℓ] = fr_flow

    # branch apparent power limits - to bus
    ytf_abs = ytf_re[ℓ]^2 + ytf_im[ℓ]^2
    ytt_abs = ytt_re[ℓ]^2 + ytt_im[ℓ]^2
    yre_to =   ytf_re[ℓ] * ytt_re[ℓ] + ytf_im[ℓ] * ytt_im[ℓ]
    yim_to = - ytf_re[ℓ] * ytt_im[ℓ] + ytf_im[ℓ] * ytt_re[ℓ]

    to_flow = vmag[to_bus]^2 * (
        ytf_abs * vmag[fr_bus]^2 + ytt_abs * vmag[to_bus]^2 +
        2.0 * vmag[fr_bus] * vmag[to_bus] * (yre_to * cosθ - yim_to * sinθ)
    )
    slines[ℓ + nlines] = to_flow
end

KA.@kernel function adj_branch_flow_kernel!(
        adj_slines, vmag, adj_vmag, vang, adj_vang,
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

    # not needed in the reverse run
    # fr_flow = vmag[fr_bus]^2 * (
    #     yff_abs * vmag[fr_bus]^2 + yft_abs * vmag[to_bus]^2 +
    #     2 * vmag[fr_bus] * vmag[to_bus] * (yre_fr * cosθ - yim_fr * sinθ)
    # )
    # slines[ℓ] = fr_flow

    # branch apparent power limits - to bus
    ytf_abs = ytf_re[ℓ]^2 + ytf_im[ℓ]^2
    ytt_abs = ytt_re[ℓ]^2 + ytt_im[ℓ]^2
    yre_to =   ytf_re[ℓ] * ytt_re[ℓ] + ytf_im[ℓ] * ytt_im[ℓ]
    yim_to = - ytf_re[ℓ] * ytt_im[ℓ] + ytf_im[ℓ] * ytt_re[ℓ]

    # not needed in the reverse run
    # to_flow = vmag[to_bus]^2 * (
    #     ytf_abs * vmag[fr_bus]^2 + ytt_abs * vmag[to_bus]^2 +
    #     2 * vmag[fr_bus] * vmag[to_bus] * (yre_to * cosθ - yim_to * sinθ)
    # )
    # slines[ℓ + nlines] = to_flow

    adj_to_flow = adj_slines[ℓ + nlines]
    adj_vmag[to_bus] += (2.0 * vmag[to_bus] * ytf_abs * vmag[fr_bus]^2
                      + 4.0 * vmag[to_bus]^3 * ytt_abs
                      + 6.0 * vmag[fr_bus] * vmag[to_bus]^2 * (yre_to * cosθ - yim_to * sinθ)
                       ) * adj_to_flow
    adj_vmag[fr_bus] += (2.0 * vmag[to_bus]^2 * vmag[fr_bus] * ytf_abs
                      + 2.0 * vmag[to_bus]^3 * (yre_to * cosθ - yim_to * sinθ)
                        ) * adj_to_flow
    adj_cosθ = 2.0 * vmag[to_bus]^3 * vmag[fr_bus] *   yre_to  * adj_to_flow
    adj_sinθ = 2.0 * vmag[to_bus]^3 * vmag[fr_bus] * (-yim_to) * adj_to_flow

    adj_from_flow = adj_slines[ℓ]
    adj_vmag[fr_bus] += (4.0 * yff_abs * vmag[fr_bus]^3
                      + 2.0 * vmag[to_bus]^2 * vmag[fr_bus] * yft_abs
                      + 6.0 * vmag[fr_bus]^2 * vmag[to_bus] * (yre_fr * cosθ - yim_fr * sinθ)
                       ) * adj_from_flow
    adj_vmag[to_bus] += (2.0 * yft_abs * vmag[fr_bus]^2 * vmag[to_bus]
                       + 2.0 * vmag[fr_bus]^3 * (yre_fr * cosθ - yim_fr * sinθ)
                        ) * adj_from_flow
    adj_cosθ += 2.0 * vmag[to_bus] * vmag[fr_bus]^3 *   yre_fr  * adj_from_flow
    adj_sinθ += 2.0 * vmag[to_bus] * vmag[fr_bus]^3 * (-yim_fr) * adj_from_flow

    adj_Δθ =   cosθ * adj_sinθ
    adj_Δθ -=  sinθ * adj_cosθ
    adj_vang[fr_bus] += adj_Δθ
    adj_vang[to_bus] -= adj_Δθ
end

