import KernelAbstractions: @index

# Implement kernels for polar formulation

"""
    function residual_kernel!(F, vmag, vang,
                              colptr, rowval,
                              ybus_re_nzval, ybus_im_nzval,
                              pnet, qload, pv, pq, nbus)

The residual CPU/GPU kernel of the powerflow residual.
"""
KA.@kernel function residual_kernel!(
    F, @Const(vmag), @Const(vang),
    @Const(colptr), @Const(rowval),
    @Const(ybus_re_nzval), @Const(ybus_im_nzval), @Const(transperm),
    @Const(pnet), @Const(pload), @Const(qload), @Const(pv), @Const(pq), nbus
)

    npv = size(pv, 1)
    npq = size(pq, 1)

    i, j = @index(Global, NTuple)
    # REAL PV: 1:npv
    # REAL PQ: (npv+1:npv+npq)
    # IMAG PQ: (npv+npq+1:npv+2npq)
    fr = (i <= npv) ? pv[i] : pq[i - npv]
    F[i, j] = -(pnet[fr, j] - pload[fr, j])
    if i > npv
        F[i + npq, j] = qload[fr, j]
    end
    @inbounds for c in colptr[fr]:colptr[fr+1]-1
        to = rowval[c]
        aij = vang[fr, j] - vang[to, j]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = vmag[fr, j]*vmag[to, j]*ybus_re_nzval[transperm[c]]
        coef_sin = vmag[fr, j]*vmag[to, j]*ybus_im_nzval[transperm[c]]
        cos_val = cos(aij)
        sin_val = sin(aij)
        F[i, j] += coef_cos * cos_val + coef_sin * sin_val
        if i > npv
            F[npq + i, j] += coef_cos * sin_val - coef_sin * cos_val
        end
    end
end

"""
    function adj_residual_edge_kernel!(
        F, adj_F, vmag, adj_vm, vang, adj_va,
        colptr, rowval,
        ybus_re_nzval, ybus_im_nzval,
        edge_vm_from, edge_vm_to,
        edge_va_from, edge_va_to,
        pnet, adj_pnet, qnet, pv, pq
    )

This kernel computes the adjoint of the voltage magnitude `adj_vm`
and `adj_va` with respect to the residual `F` and the adjoint `adj_F`.

To avoid a race condition, each thread sums its contribution on the edge of the network graph.
"""
KA.@kernel function adj_residual_edge_kernel!(
    F, @Const(adj_F), @Const(vmag), adj_vm, vang, adj_va,
    @Const(colptr), @Const(rowval),
    @Const(ybus_re_nzval), @Const(ybus_im_nzval), @Const(transperm),
    edge_vm_from, edge_vm_to,
    edge_va_from, edge_va_to,
    @Const(pnet), adj_pnet, @Const(pload), @Const(qload), @Const(pv), @Const(pq)
)

    npv = size(pv, 1)
    npq = size(pq, 1)

    i, j = @index(Global, NTuple)
    # REAL PV: 1:npv
    # REAL PQ: (npv+1:npv+npq)
    # IMAG PQ: (npv+npq+1:npv+2npq)
    fr = (i <= npv) ? pv[i] : pq[i - npv]
    F[i, j] = -(pnet[fr] - pload[fr])
    if i > npv
        F[i + npq, j] = qload[fr]
    end
    @inbounds for c in colptr[fr]:colptr[fr+1]-1
        # Forward loop
        to = rowval[c]
        aij = vang[fr, j] - vang[to, j]

        yre = ybus_re_nzval[transperm[c]]
        yim = ybus_im_nzval[transperm[c]]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = vmag[fr, j]*vmag[to, j]*yre
        coef_sin = vmag[fr, j]*vmag[to, j]*yim

        cos_val = cos(aij)
        sin_val = sin(aij)
        F[i, j] += coef_cos * cos_val + coef_sin * sin_val
        if i > npv
            F[npq + i, j] += coef_cos * sin_val - coef_sin * cos_val
        end

        adj_coef_cos =  cos_val  * adj_F[i]
        adj_coef_sin =  sin_val  * adj_F[i]
        adj_cos_val  =  coef_cos * adj_F[i]
        adj_sin_val  =  coef_sin * adj_F[i]

        if i > npv
            adj_coef_cos +=  sin_val  * adj_F[npq + i]
            adj_coef_sin += -cos_val  * adj_F[npq + i]
            adj_cos_val  += -coef_sin * adj_F[npq + i]
            adj_sin_val  +=  coef_cos * adj_F[npq + i]
        end

        adj_aij =   cos_val*adj_sin_val
        adj_aij += -sin_val*adj_cos_val

        edge_vm_from[c, j] += vmag[to, j]*yim*adj_coef_sin
        edge_vm_to[c, j]   += vmag[fr, j]*yim*adj_coef_sin
        edge_vm_from[c, j] += vmag[to, j]*yre*adj_coef_cos
        edge_vm_to[c, j]   += vmag[fr, j]*yre*adj_coef_cos

        edge_va_from[c, j] += adj_aij
        edge_va_to[c, j]   -= adj_aij
    end
    # qnet is not active
    # if i > npv
    #     adj_qnet[fr] -= adj_F[i + npq]
    # end
    adj_pnet[fr, j] -= adj_F[i]
end

"""
    function cpu_adj_node_kernel!(F, adj_F, vmag, adj_vm, vang, adj_va,
                                  colptr, rowval,
                                  edge_vm_from, edge_vm_to,
                                  edge_va_from, edge_va_to
    )

This kernel accumulates the adjoint of the voltage magnitude `adj_vm`
and `adj_va` from the edges of the graph stored as CSC matrices.
"""
function cpu_adj_node_kernel!(
    adj_vm, adj_va,
    colptr, rowval,
    edge_vm_from, edge_vm_to,
    edge_va_from, edge_va_to, dest,
)
    for i in 1:size(adj_vm, 1), j in 1:size(adj_vm, 2)
        @inbounds for c in colptr[i]:colptr[i+1]-1
            adj_vm[i, j] += edge_vm_from[c, j]
            adj_vm[i, j] += edge_vm_to[dest[c], j]
            adj_va[i, j] += edge_va_from[c, j]
            adj_va[i, j] += edge_va_to[dest[c], j]
        end
    end
end

"""
    function gpu_adj_node_kernel!(adj_vm, adj_va,
                                  colptr, rowval,
                                  edge_vm_from, edge_va_from,
                                  edge_vm_to, edge_va_to, perm,
    )

This kernel accumulates the adjoint of the voltage magnitude `adj_vm`
and `adj_va` from the edges of the graph. For the `to` edges a COO matrix
was used to compute the transposed of the graph to add them to the `from` edges.
The permutation corresponding to the transpose operation is stored inplace,
in vector `perm`.

"""
KA.@kernel function gpu_adj_node_kernel!(
    adj_vm, adj_va,
    @Const(colptr), @Const(rowval),
    @Const(edge_vm_from), @Const(edge_vm_to),
    @Const(edge_va_from), @Const(edge_va_to), @Const(dest)
)
    i, j = @index(Global, NTuple)
    @inbounds for c in colptr[i]:colptr[i+1]-1
        to = dest[c]
        adj_vm[i, j] += edge_vm_from[c, j]
        adj_vm[i, j] += edge_vm_to[to, j]
        adj_va[i, j] += edge_va_from[c, j]
        adj_va[i, j] += edge_va_to[to, j]
    end
end

"""
    function adj_residual_polar!(
        F, adj_F, vmag, adj_vm, vang, adj_va,
        ybus_re, ybus_im,
        pnet, adj_pnet, qnet,
        edge_vm_from, edge_vm_to, edge_va_from, edge_va_to,
        pv, pq, nbus
    ) where {T}

This is the wrapper of the adjoint kernel that computes the adjoint of
the voltage magnitude `adj_vm` and `adj_va` with respect to the residual `F`
and the adjoint `adj_F`.
"""
function adj_residual_polar!(
    F, adj_F, vmag, adj_vm, vang, adj_va,
    ybus_re, ybus_im, transpose_perm,
    pnet, adj_pnet, pload, qload,
    edge_vm_from, edge_vm_to, edge_va_from, edge_va_to,
    pv, pq, nbus, device
)
    npv = length(pv)
    npq = length(pq)
    nvbus = size(vmag, 1)
    nnz = length(ybus_re.nzval)
    colptr = ybus_re.colptr
    rowval = ybus_re.rowval

    kernel_edge! = adj_residual_edge_kernel!(device)
    ev = kernel_edge!(F, adj_F, vmag, adj_vm, vang, adj_va,
                 ybus_re.colptr, ybus_re.rowval,
                 ybus_re.nzval, ybus_im.nzval, transpose_perm,
                 edge_vm_from, edge_vm_to,
                 edge_va_from, edge_va_to,
                 pnet, adj_pnet, pload, qload, pv, pq,
                 ndrange=(npv+npq, size(F, 2)),
                 dependencies = Event(device)
    )
    wait(ev)

    # The permutation corresponding to the transpose of Ybus.
    # is given in transpose_perm
    if isa(device, CPU)
        cpu_adj_node_kernel!(
            adj_vm, adj_va,
            ybus_re.colptr, ybus_re.rowval,
            edge_vm_from, edge_vm_to,
            edge_va_from, edge_va_to, transpose_perm,
        )
    else
        ev = gpu_adj_node_kernel!(device)(
            adj_vm, adj_va,
            ybus_re.colptr, ybus_re.rowval,
            edge_vm_from, edge_vm_to,
            edge_va_from, edge_va_to, transpose_perm,
            ndrange=(nvbus, size(adj_vm, 2)),
            dependencies = Event(device)
        )
        wait(ev)
    end
end

@inline function bus_injection(
    bus, j, vmag, vang, ybus_re_colptr, ybus_re_rowval, ybus_re_nzval, ybus_im_nzval, transperm,
)
    inj = 0.0
    @inbounds for c in ybus_re_colptr[bus]:ybus_re_colptr[bus+1]-1
        to = ybus_re_rowval[c]
        aij = vang[bus, j] - vang[to, j]
        coef_cos = vmag[bus, j]*vmag[to, j]*ybus_re_nzval[transperm[c]]
        coef_sin = vmag[bus, j]*vmag[to, j]*ybus_im_nzval[transperm[c]]
        cos_val = cos(aij)
        sin_val = sin(aij)
        inj += coef_cos * cos_val + coef_sin * sin_val
    end
    return inj
end

@inline function adjoint_bus_injection!(
    fr, j, adj_inj, adj_vmag, adj_vang, vmag, vang,
    ybus_re_colptr, ybus_re_rowval, ybus_re_nzval, ybus_im_nzval, transperm,
)
    @inbounds for c in ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1
        to = ybus_re_rowval[c]
        aij = vang[fr, j] - vang[to, j]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        yre = ybus_re_nzval[transperm[c]]
        yim = ybus_im_nzval[transperm[c]]
        coef_cos = vmag[fr, j]*vmag[to, j]*yre
        coef_sin = vmag[fr, j]*vmag[to, j]*yim
        cosθ = cos(aij)
        sinθ = sin(aij)

        adj_coef_cos = cosθ  * adj_inj
        adj_cos_val  = coef_cos * adj_inj
        adj_coef_sin = sinθ  * adj_inj
        adj_sin_val  = coef_sin * adj_inj

        adj_aij =   cosθ * adj_sin_val
        adj_aij -=  sinθ * adj_cos_val

        adj_vmag[fr, j] += vmag[to, j] * yre * adj_coef_cos
        adj_vmag[to, j] += vmag[fr, j] * yre * adj_coef_cos
        adj_vmag[fr, j] += vmag[to, j] * yim * adj_coef_sin
        adj_vmag[to, j] += vmag[fr, j] * yim * adj_coef_sin

        adj_vang[fr, j] += adj_aij
        adj_vang[to, j] -= adj_aij
    end
end

KA.@kernel function transfer_kernel!(
    vmag, vang, pnet, qnet, @Const(u), @Const(pv), @Const(pq), @Const(ref), @Const(pload), @Const(qload)
)
    i, j = @index(Global, NTuple)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    # PV bus
    if i <= npv
        bus = pv[i]
        vmag[bus, j] = u[nref + i, j]
        pnet[bus, j] = u[nref + npv + i, j]
    # REF bus
    else
        i_ref = i - npv
        bus = ref[i_ref]
        vmag[bus, j] = u[i_ref, j]
    end
end

# Transfer values in (x, u) to buffer
function transfer!(polar::PolarForm, buffer::PolarNetworkState, u)
    kernel! = transfer_kernel!(polar.device)
    nbus = length(buffer.vmag)
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())
    ndrange = (length(pv)+length(ref), size(u, 2))
    ev = kernel!(
        buffer.vmag, buffer.vang, buffer.pnet, buffer.qnet,
        u,
        pv, pq, ref,
        buffer.pload, buffer.qload,
        ndrange=ndrange,
        dependencies = Event(polar.device)
    )
    wait(ev)
end

KA.@kernel function adj_transfer_kernel!(
    adj_u, adj_x, @Const(adj_vmag), @Const(adj_vang), @Const(adj_pnet), @Const(pv), @Const(pq), @Const(ref),
)
    i, j = @index(Global, NTuple)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    # PQ buses
    if i <= npq
        bus = pq[i]
        adj_x[npv+i, j] =  adj_vang[bus, j]
        adj_x[npv+npq+i, j] = adj_vmag[bus, j]
    # PV buses
    elseif i <= npq + npv
        i_ = i - npq
        bus = pv[i_]
        adj_u[nref + i_, j] = adj_vmag[bus, j]
        adj_u[nref + npv + i_, j] = adj_pnet[bus, j]
        adj_x[i_, j] = adj_vang[bus, j]
    # SLACK buses
    elseif i <= npq + npv + nref
        i_ = i - npq - npv
        bus = ref[i_]
        adj_u[i_, j] = adj_vmag[bus, j]
    end
end

# Transfer values in (x, u) to buffer
function adjoint_transfer!(
    polar::PolarForm,
    ∂u, ∂x,
    ∂vmag, ∂vang, ∂pnet,
)
    nbus = get(polar, PS.NumberOfBuses())
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    ev = adj_transfer_kernel!(polar.device)(
        ∂u, ∂x,
        ∂vmag, ∂vang, ∂pnet,
        pv, pq, ref;
        ndrange=(nbus, size(∂u, 2)),
        dependencies=Event(polar.device)
    )
    wait(ev)
end

KA.@kernel function _reverse_transfer_kernel2!(
        output, @Const(adj_vmag), @Const(adj_vang), @Const(adj_pgen), @Const(pv), @Const(pq), @Const(ref), @Const(pv2gen),
    npq, npv, nref, ngen,
)
    i, j = @index(Global, NTuple)

    output[i, j] = if i <= npv
        # x (vang_pv)
        k = pv[i]
        adj_vang[k, j]
    elseif i <= npv + npq
        k = pq[i - npv]
        # x (vang_pq)
        adj_vang[k, j]
    elseif i <= npv + 2*npq
        # x (vmag_pq)
        k = pq[i - npv - npq]
        adj_vmag[k, j]
    elseif i <= npv + 2*npq + nref
        # u (vmag_ref)
        k = ref[i - npv - 2*npq]
        adj_vmag[k, j]
    elseif i <= npv + 2*npq + nref + npv
        # u (vmag_pv)
        k = pv[i - npv - 2*npq - nref]
        adj_vmag[k, j]
    elseif i <= npv + 2*npq + nref + npv + ngen
        # u (vmag_pg)
        k = pv2gen[i - 2*npv - 2*npq - nref]
        adj_pgen[k, j]
    end

end

function reverse_transfer!(
    polar::PolarForm,
    output, ∂state,
)
    nx = get(polar, ExaPF.NumberOfState())
    nu = get(polar, ExaPF.NumberOfControl())
    nbus = get(polar, PS.NumberOfBuses())
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    pv2gen = polar.indexing.index_pv_to_gen
    ev = _reverse_transfer_kernel2!(polar.device)(
        output,
        ∂state.vmag, ∂state.vang, ∂state.pgen,
        pv, pq, ref, pv2gen,
        length(pq), length(pv), length(ref), length(pv2gen),
        ndrange=(nx+nu, size(output, 2)),
        dependencies=Event(polar.device)
    )
    wait(ev)
end

KA.@kernel function active_power_slack!(
    cons, vmag, vang, ref, pd,
    @Const(ybus_re_nzval), @Const(ybus_re_colptr), @Const(ybus_re_rowval), @Const(ybus_im_nzval),
    @Const(transperm),
)
    i = @index(Global, Linear)
    bus = ref[i]
    inj = 0.0
    @inbounds for c in ybus_re_colptr[bus]:ybus_re_colptr[bus+1]-1
        to = ybus_re_rowval[c]
        aij = vang[bus] - vang[to]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = vmag[bus]*vmag[to]*ybus_re_nzval[transperm[c]]
        coef_sin = vmag[bus]*vmag[to]*ybus_im_nzval[transperm[c]]
        cos_val = cos(aij)
        sin_val = sin(aij)
        inj += coef_cos * cos_val + coef_sin * sin_val
    end
    cons[i] = inj + pd[bus]
end

KA.@kernel function adj_active_power_slack!(
    v_m, v_a, adj_v_m, adj_v_a, adj_P, ref,
    @Const(ybus_re_nzval), @Const(ybus_re_colptr), @Const(ybus_re_rowval), @Const(ybus_im_nzval),
    @Const(transperm),
)
    i = @index(Global, Linear)
    fr = ref[i]
    @inbounds for c in ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1
        to = ybus_re_rowval[c]
        aij = v_a[fr] - v_a[to]
        cosθ = cos(aij)
        sinθ = sin(aij)

        yre = ybus_re_nzval[transperm[c]]
        yim = ybus_im_nzval[transperm[c]]

        cθ = yre*cosθ
        sθ = yim*sinθ
        adj_v_m[fr] += v_m[to] * (cθ + sθ) * adj_P[i]
        adj_v_m[to] += v_m[fr] * (cθ + sθ) * adj_P[i]

        adj_aij = -(v_m[fr]*v_m[to]*(yre*sinθ))
        adj_aij += v_m[fr]*v_m[to]*(yim*cosθ)
        adj_aij *= adj_P[i]
        adj_v_a[to] += -adj_aij
        adj_v_a[fr] += adj_aij
    end
end

KA.@kernel function reactive_power_kernel!(
    qg, @Const(vmag), @Const(vang), @Const(pnet),
    @Const(pv), @Const(ref), @Const(pv_to_gen), @Const(ref_to_gen),
    @Const(ybus_re_nzval), @Const(ybus_re_colptr), @Const(ybus_re_rowval),
    @Const(ybus_im_nzval), @Const(transperm), @Const(qload)
)
    i, j = @index(Global, NTuple)
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
        aij = vang[bus, j] - vang[to, j]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = vmag[bus, j]*vmag[to, j]*ybus_re_nzval[transperm[c]]
        coef_sin = vmag[bus, j]*vmag[to, j]*ybus_im_nzval[transperm[c]]
        cos_val = cos(aij)
        sin_val = sin(aij)
        inj += coef_cos * sin_val - coef_sin * cos_val
    end
    qg[i_gen, j] = inj + qload[bus]
end

KA.@kernel function adj_reactive_power_edge_kernel!(
    qg, adj_qg,
    @Const(vmag), adj_vmag, @Const(vang), adj_vang,
    @Const(pnet), adj_pnet,
    @Const(pv), @Const(ref), @Const(pv_to_gen), @Const(ref_to_gen),
    edge_vmag_bus, edge_vmag_to,
    edge_vang_bus, edge_vang_to,
    @Const(ybus_re_nzval), @Const(ybus_re_colptr), @Const(ybus_re_rowval),
    @Const(ybus_im_nzval), @Const(transperm), @Const(qload)
)
    i, j = @index(Global, NTuple)
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
        aij = vang[bus, j] - vang[to, j]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = vmag[bus, j]*vmag[to, j]*ybus_re_nzval[transperm[c]]
        coef_sin = vmag[bus, j]*vmag[to, j]*ybus_im_nzval[transperm[c]]
        cos_val = cos(aij)
        sin_val = sin(aij)
        inj += coef_cos * sin_val - coef_sin * cos_val
    end
    qg[i_gen, j] = inj + qload[bus]

    # Reverse run
    adj_inj = adj_qg[i_gen, j]
    @inbounds for c in ybus_re_colptr[bus]:ybus_re_colptr[bus+1]-1
        to = ybus_re_rowval[c]
        aij = vang[bus, j] - vang[to, j]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        yre = ybus_re_nzval[transperm[c]]
        yim = ybus_im_nzval[transperm[c]]
        coef_cos = vmag[bus, j]*vmag[to, j]*yre
        coef_sin = vmag[bus, j]*vmag[to, j]*yim
        cos_val = cos(aij)
        sin_val = sin(aij)

        adj_coef_cos = sin_val  * adj_inj
        adj_sin_val  = coef_cos * adj_inj
        adj_coef_sin = -cos_val  * adj_inj
        adj_cos_val  = -coef_sin * adj_inj

        adj_aij =   coef_cos * cos_val * adj_inj
        adj_aij +=  coef_sin * sin_val * adj_inj

        edge_vmag_bus[c, j] += vmag[to, j] *yre*adj_coef_cos
        edge_vmag_to[c, j]  += vmag[bus, j]*yre*adj_coef_cos
        edge_vmag_bus[c, j] += vmag[to, j] *yim*adj_coef_sin
        edge_vmag_to[c, j]  += vmag[bus, j]*yim*adj_coef_sin

        edge_vang_bus[c, j] += adj_aij
        edge_vang_to[c, j]  -= adj_aij
    end
end

function adj_reactive_power!(
    F, adj_F, vmag, adj_vm, vang, adj_va,
    ybus_re, ybus_im, transpose_perm,
    pnet, adj_pnet,
    edge_vm_from, edge_vm_to, edge_va_from, edge_va_to,
    reactive_load,
    pv, pq, ref, pv_to_gen, ref_to_gen, nbus, device
)
    npv = length(pv)
    npq = length(pq)
    nvbus = length(vmag)
    nnz = length(ybus_re.nzval)

    colptr = ybus_re.colptr
    rowval = ybus_re.rowval

    kernel_edge! = adj_reactive_power_edge_kernel!(device)

    ndrange = (length(pv) + length(ref), size(adj_F, 2))

    ev = kernel_edge!(
        F, adj_F,
        vmag, adj_vm,
        vang, adj_va,
        pnet, adj_pnet,
        pv, ref, pv_to_gen, ref_to_gen,
        edge_vm_from, edge_vm_to,
        edge_va_from, edge_va_to,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
        ybus_im.nzval, transpose_perm, reactive_load,
        ndrange=ndrange,
        dependencies=Event(device)
    )
    wait(ev)

    if isa(device, CPU)
        cpu_adj_node_kernel!(
            adj_vm, adj_va,
            ybus_re.colptr, ybus_re.rowval,
            edge_vm_from, edge_vm_to,
            edge_va_from, edge_va_to, transpose_perm,
        )
    else
        ev = gpu_adj_node_kernel!(device)(
            adj_vm, adj_va,
            ybus_re.colptr, ybus_re.rowval,
            edge_vm_from, edge_vm_to,
            edge_va_from, edge_va_to, transpose_perm,
            ndrange=(nvbus, size(adj_vm, 2)),
            dependencies=Event(device)
        )
        wait(ev)
    end

end

KA.@kernel function branch_flow_kernel!(
    slines, @Const(vmag), @Const(vang),
    @Const(yff_re), @Const(yft_re), @Const(ytf_re), @Const(ytt_re),
    @Const(yff_im), @Const(yft_im), @Const(ytf_im), @Const(ytt_im),
    @Const(f), @Const(t), nlines,
)
    ℓ, j = @index(Global, NTuple)
    fr_bus = f[ℓ]
    to_bus = t[ℓ]

    Δθ = vang[fr_bus, j] - vang[to_bus, j]
    cosθ = cos(Δθ)
    sinθ = sin(Δθ)

    # branch apparent power limits - from bus
    yff_abs = yff_re[ℓ]^2 + yff_im[ℓ]^2
    yft_abs = yft_re[ℓ]^2 + yft_im[ℓ]^2
    yre_fr =   yff_re[ℓ] * yft_re[ℓ] + yff_im[ℓ] * yft_im[ℓ]
    yim_fr = - yff_re[ℓ] * yft_im[ℓ] + yff_im[ℓ] * yft_re[ℓ]

    fr_flow = vmag[fr_bus, j]^2 * (
        yff_abs * vmag[fr_bus, j]^2 + yft_abs * vmag[to_bus, j]^2 +
        2.0 * vmag[fr_bus, j] * vmag[to_bus, j] * (yre_fr * cosθ - yim_fr * sinθ)
    )
    slines[ℓ, j] = fr_flow

    # branch apparent power limits - to bus
    ytf_abs = ytf_re[ℓ]^2 + ytf_im[ℓ]^2
    ytt_abs = ytt_re[ℓ]^2 + ytt_im[ℓ]^2
    yre_to =   ytf_re[ℓ] * ytt_re[ℓ] + ytf_im[ℓ] * ytt_im[ℓ]
    yim_to = - ytf_re[ℓ] * ytt_im[ℓ] + ytf_im[ℓ] * ytt_re[ℓ]

    to_flow = vmag[to_bus, j]^2 * (
        ytf_abs * vmag[fr_bus, j]^2 + ytt_abs * vmag[to_bus, j]^2 +
        2.0 * vmag[fr_bus, j] * vmag[to_bus, j] * (yre_to * cosθ - yim_to * sinθ)
    )
    slines[ℓ + nlines, j] = to_flow
end

KA.@kernel function adj_branch_flow_edge_kernel!(
    @Const(adj_slines), @Const(vmag), @Const(adj_vmag), @Const(vang), @Const(adj_vang),
    adj_va_to_lines, adj_va_from_lines, adj_vm_to_lines, adj_vm_from_lines,
    @Const(yff_re), @Const(yft_re), @Const(ytf_re), @Const(ytt_re),
    @Const(yff_im), @Const(yft_im), @Const(ytf_im), @Const(ytt_im),
    @Const(f), @Const(t), nlines,
)
    ℓ, j = @index(Global, NTuple)
    fr_bus = f[ℓ]
    to_bus = t[ℓ]

    Δθ = vang[fr_bus, j] - vang[to_bus, j]
    cosθ = cos(Δθ)
    sinθ = sin(Δθ)
    dcosθ = -sinθ
    dsinθ = cosθ

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

    adj_to_flow = adj_slines[ℓ + nlines, j]
    adj_vm_to_lines[ℓ, j] += (2.0 * vmag[to_bus, j] * ytf_abs * vmag[fr_bus, j]^2
                      + 4.0 * vmag[to_bus, j]^3 * ytt_abs
                      + 6.0 * vmag[fr_bus, j] * vmag[to_bus, j]^2 * (yre_to * cosθ - yim_to * sinθ)
                       ) * adj_to_flow
    adj_vm_from_lines[ℓ, j] += (2.0 * vmag[to_bus, j]^2 * vmag[fr_bus, j] * ytf_abs
                      + 2.0 * vmag[to_bus, j]^3 * (yre_to * cosθ - yim_to * sinθ)
                        ) * adj_to_flow
    adj_cosθ = 2.0 * vmag[to_bus, j]^3 * vmag[fr_bus, j] *   yre_to  * adj_to_flow
    adj_sinθ = 2.0 * vmag[to_bus, j]^3 * vmag[fr_bus, j] * (-yim_to) * adj_to_flow

    adj_from_flow = adj_slines[ℓ, j]
    adj_vm_from_lines[ℓ, j] += (4.0 * yff_abs * vmag[fr_bus, j]^3
                      + 2.0 * vmag[to_bus, j]^2 * vmag[fr_bus, j] * yft_abs
                      + 6.0 * vmag[fr_bus, j]^2 * vmag[to_bus, j] * (yre_fr * cosθ - yim_fr * sinθ)
                       ) * adj_from_flow
    adj_vm_to_lines[ℓ, j] += (2.0 * yft_abs * vmag[fr_bus, j]^2 * vmag[to_bus, j]
                       + 2.0 * vmag[fr_bus, j]^3 * (yre_fr * cosθ - yim_fr * sinθ)
                        ) * adj_from_flow
    adj_cosθ += 2.0 * vmag[to_bus, j] * vmag[fr_bus, j]^3 *   yre_fr  * adj_from_flow
    adj_sinθ += 2.0 * vmag[to_bus, j] * vmag[fr_bus, j]^3 * (-yim_fr) * adj_from_flow

    adj_Δθ =   dsinθ * adj_sinθ
    adj_Δθ +=  dcosθ * adj_cosθ
    adj_va_from_lines[ℓ, j] += adj_Δθ
    adj_va_to_lines[ℓ, j] -= adj_Δθ
end

function adj_branch_flow!(
        adj_slines, vmag, adj_vm, vang, adj_va,
        adj_vm_from_lines, adj_va_from_lines, adj_vm_to_lines, adj_va_to_lines,
        yff_re, yft_re, ytf_re, ytt_re,
        yff_im, yft_im, ytf_im, ytt_im,
        f, t, Cf, Ct, nlines, device
    )
    nvbus = length(vang)

    ev = adj_branch_flow_edge_kernel!(device)(
            adj_slines, vmag, adj_vm, vang, adj_va,
            adj_va_to_lines, adj_va_from_lines, adj_vm_to_lines, adj_vm_from_lines,
            yff_re, yft_re, ytf_re, ytt_re,
            yff_im, yft_im, ytf_im, ytt_im,
            f, t, nlines, ndrange = (nlines, size(adj_slines, 2)),
            dependencies=Event(device)
    )
    wait(ev)

    # Aggregate the adjoints on the nodes using the bus-node adjacency matrices.
    # mul! should be overloaded on the GPU to work with dual numbers
    # (needed to evaluate the Hessian using forward over reverse)
    mul!(adj_vm, Cf, adj_vm_from_lines, 1.0, 1.0)
    mul!(adj_vm, Ct, adj_vm_to_lines, 1.0, 1.0)
    mul!(adj_va, Cf, adj_va_from_lines, 1.0, 1.0)
    mul!(adj_va, Ct, adj_va_to_lines, 1.0, 1.0)
end

KA.@kernel function basis_kernel!(
    cons, @Const(vmag), @Const(vang), @Const(f), @Const(t), nlines, nbus,
)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if i <= nlines
            ℓ = i
            fr_bus = f[ℓ]
            to_bus = t[ℓ]
            Δθ = vang[fr_bus, j] - vang[to_bus, j]
            cosθ = cos(Δθ)
            cons[i, j] = vmag[fr_bus, j] * vmag[to_bus, j] * cosθ
        elseif i <= 2 * nlines
            ℓ = i - nlines
            fr_bus = f[ℓ]
            to_bus = t[ℓ]
            Δθ = vang[fr_bus, j] - vang[to_bus, j]
            sinθ = sin(Δθ)
            cons[i, j] = vmag[fr_bus, j] * vmag[to_bus, j] * sinθ
        elseif i <= 2 * nlines + nbus
            b = i - 2 * nlines
            cons[i, j] = vmag[b, j] * vmag[b, j]
        end
    end
end

KA.@kernel function adj_basis_kernel!(
    ∂cons, adj_vmag, adj_vmag_fr, adj_vmag_to,
    adj_vang_fr, adj_vang_to,
    @Const(vmag), @Const(vang), @Const(f), @Const(t), nlines, nbus,
)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if i <= nlines
            ℓ = i
            fr_bus = f[ℓ]
            to_bus = t[ℓ]
            Δθ = vang[fr_bus, j] - vang[to_bus, j]
            cosθ = cos(Δθ)
            sinθ = sin(Δθ)

            adj_vang_fr[i] += -vmag[fr_bus, j] * vmag[to_bus, j] * sinθ * ∂cons[ℓ, j]
            adj_vang_fr[i] +=  vmag[fr_bus, j] * vmag[to_bus, j] * cosθ * ∂cons[ℓ+nlines, j]
            adj_vang_to[i] +=  vmag[fr_bus, j] * vmag[to_bus, j] * sinθ * ∂cons[ℓ, j]
            adj_vang_to[i] -=  vmag[fr_bus, j] * vmag[to_bus, j] * cosθ * ∂cons[ℓ+nlines, j]

            adj_vmag_fr[i] +=  vmag[to_bus, j] * cosθ * ∂cons[ℓ, j]
            adj_vmag_fr[i] += vmag[to_bus, j] * sinθ * ∂cons[ℓ+nlines, j]

            adj_vmag_to[i] +=  vmag[fr_bus, j] * cosθ * ∂cons[ℓ, j]
            adj_vmag_to[i] += vmag[fr_bus, j] * sinθ * ∂cons[ℓ+nlines, j]
        else i <= nlines + nbus
            b = i - nlines
            adj_vmag[b, j] += 2.0 * vmag[b, j] * ∂cons[b+2*nlines, j]
        end
    end
end
