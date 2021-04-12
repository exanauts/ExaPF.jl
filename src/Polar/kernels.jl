import KernelAbstractions: @index

# Implement kernels for polar formulation

"""
    function residual_kernel!(F, vm, va,
                              colptr, rowval,
                              ybus_re_nzval, ybus_im_nzval,
                              pinj, qload, pv, pq, nbus)

The residual CPU/GPU kernel of the powerflow residual.
"""
KA.@kernel function residual_kernel!(
    F, vm, va,
    colptr, rowval,
    ybus_re_nzval, ybus_im_nzval,
    pinj, qload, pv, pq, nbus
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
        F[i + npq] -= qload[fr]
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

"""
    function adj_residual_edge_kernel!(
        F, adj_F, vm, adj_vm, va, adj_va,
        colptr, rowval,
        ybus_re_nzval, ybus_im_nzval,
        edge_vm_from, edge_vm_to,
        edge_va_from, edge_va_to,
        pinj, adj_pinj, qinj, pv, pq
    )

This kernel computes the adjoint of the voltage magnitude `adj_vm`
and `adj_va` with respect to the residual `F` and the adjoint `adj_F`.

To avoid a race condition, each thread sums its contribution on the edge of the network graph.
"""
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

"""
    function cpu_adj_node_kernel!(F, adj_F, vm, adj_vm, va, adj_va,
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
    for i in 1:length(adj_vm)
        @inbounds for c in colptr[i]:colptr[i+1]-1
            adj_vm[i] += edge_vm_from[c]
            adj_vm[i] += edge_vm_to[dest[c]]
            adj_va[i] += edge_va_from[c]
            adj_va[i] += edge_va_to[dest[c]]
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
    colptr, rowval,
    edge_vm_from, edge_vm_to,
    edge_va_from, edge_va_to, dest
)
    i = @index(Global, Linear)
    @inbounds for c in colptr[i]:colptr[i+1]-1
        to = dest[c]
        adj_vm[i] += edge_vm_from[c]
        adj_vm[i] += edge_vm_to[to]
        adj_va[i] += edge_va_from[c]
        adj_va[i] += edge_va_to[to]
    end
end

"""
    function adj_residual_polar!(
        F, adj_F, vm, adj_vm, va, adj_va,
        ybus_re, ybus_im,
        pinj, adj_pinj, qinj,
        edge_vm_from, edge_vm_to, edge_va_from, edge_va_to,
        pv, pq, nbus
    ) where {T}

This is the wrapper of the adjoint kernel that computes the adjoint of
the voltage magnitude `adj_vm` and `adj_va` with respect to the residual `F`
and the adjoint `adj_F`.
"""
function adj_residual_polar!(
    F, adj_F, vm, adj_vm, va, adj_va,
    ybus_re, ybus_im, transpose_perm,
    pinj, adj_pinj, qinj,
    edge_vm_from, edge_vm_to, edge_va_from, edge_va_to,
    pv, pq, nbus, device
)
    npv = length(pv)
    npq = length(pq)
    nvbus = length(vm)
    nnz = length(ybus_re.nzval)
    colptr = ybus_re.colptr
    rowval = ybus_re.rowval

    kernel_edge! = adj_residual_edge_kernel!(device)
    ev = kernel_edge!(F, adj_F, vm, adj_vm, va, adj_va,
                 ybus_re.colptr, ybus_re.rowval,
                 ybus_re.nzval, ybus_im.nzval,
                 edge_vm_from, edge_vm_to,
                 edge_va_from, edge_va_to,
                 pinj, adj_pinj, qinj, pv, pq,
                 ndrange=npv+npq,
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
            ndrange=nvbus,
            dependencies = Event(device)
        )
        wait(ev)
    end
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
    ev = kernel!(
        buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj,
        u,
        pv, pq, ref,
        polar.active_load, polar.reactive_load,
        ndrange=(length(pv)+length(ref)),
        dependencies = Event(polar.device)
    )
    wait(ev)
end

KA.@kernel function adj_transfer_kernel!(
    adj_u, adj_x, adj_vmag, adj_vang, adj_pinj, pv, pq, ref,
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

function _cpu_adj_transfer!(
    adj_u, adj_x, adj_vmag, adj_vang, adj_pinj, pv, pq, ref,
)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    # PQ buses
    @inbounds for i in 1:npq
        bus = pq[i]
        adj_x[npv+i] =  adj_vang[bus]
        adj_x[npv+npq+i] = adj_vmag[bus]
    end
    # PV buses
    @inbounds for i_ in 1:npv
        bus = pv[i_]
        adj_u[nref + i_] = adj_vmag[bus]
        adj_u[nref + npv + i_] += adj_pinj[bus]
        adj_x[i_] = adj_vang[bus]
    end
    # SLACK buses
    @inbounds for i_ in 1:nref
        bus = ref[i_]
        adj_u[i_] = adj_vmag[bus]
    end
end

# Transfer values in (x, u) to buffer
function adjoint_transfer!(
    polar::PolarForm,
    ∂u, ∂x,
    ∂vm, ∂va, ∂pinj,
)
    nbus = get(polar, PS.NumberOfBuses())
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    # The adjoint of transfer is performance critical code.
    # To avoid KernelAbstractions's overhead on the CPU, we
    # don't use a kernel and use _cpu_adj_transfer instead.
    if isa(polar.device, CPU)
        _cpu_adj_transfer!(
            ∂u, ∂x, ∂vm, ∂va, ∂pinj, pv, pq, ref
        )
    else
        ev = adj_transfer_kernel!(polar.device)(
            ∂u, ∂x,
            ∂vm, ∂va, ∂pinj,
            pv, pq, ref;
            ndrange=nbus,
            dependencies=Event(polar.device)
        )
        wait(ev)
    end
end

KA.@kernel function active_power_kernel!(
    pg, vmag, vang, pinj,
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
    kernel! = active_power_kernel!(polar.device)
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    range_ = length(pv) + length(ref)

    ev = kernel!(
        buffer.pg,
        buffer.vmag, buffer.vang, buffer.pinj,
        pv, ref, pv_to_gen, ref_to_gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
        ybus_im.nzval, polar.active_load,
        ndrange=range_,
        dependencies=Event(polar.device)
    )
    wait(ev)
end

KA.@kernel function adj_active_power_kernel!(
    adj_pg,
    vmag, adj_vmag, vang, adj_vang, adj_pinj,
    pv, ref, pv_to_gen, ref_to_gen,
    ybus_re_nzval, ybus_re_colptr, ybus_re_rowval, ybus_im_nzval,
)
    i = @index(Global, Linear)
    npv = length(pv)
    nref = length(ref)
    if i <= npv
        bus = pv[i]
        i_gen = pv_to_gen[i]
        adj_pinj[bus] = adj_pg[i_gen]
    # Evaluate active power at slack nodes
    elseif i <= npv + nref
        i_ = i - npv
        fr = ref[i_]
        i_gen = ref_to_gen[i_]

        adj_inj = adj_pg[i_gen]
        @inbounds for c in ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1
            to = ybus_re_rowval[c]
            aij = vang[fr] - vang[to]
            # f_re = a * cos + b * sin
            # f_im = a * sin - b * cos
            coef_cos = vmag[fr]*vmag[to]*ybus_re_nzval[c]
            coef_sin = vmag[fr]*vmag[to]*ybus_im_nzval[c]
            cosθ = cos(aij)
            sinθ = sin(aij)

            adj_coef_cos = cosθ  * adj_inj
            adj_cos_val  = coef_cos * adj_inj
            adj_coef_sin = sinθ  * adj_inj
            adj_sin_val  = coef_sin * adj_inj

            adj_aij =   cosθ * adj_sin_val
            adj_aij -=  sinθ * adj_cos_val

            adj_vmag[fr] += vmag[to] * ybus_re_nzval[c] * adj_coef_cos
            adj_vmag[to] += vmag[fr] * ybus_re_nzval[c] * adj_coef_cos
            adj_vmag[fr] += vmag[to] * ybus_im_nzval[c] * adj_coef_sin
            adj_vmag[to] += vmag[fr] * ybus_im_nzval[c] * adj_coef_sin

            adj_vang[fr] += adj_aij
            adj_vang[to] -= adj_aij
        end
    end
end

KA.@kernel function reactive_power_kernel!(
    qg, vmag, vang, pinj,
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
end

function update!(polar::PolarForm, ::PS.Generators, ::PS.ReactivePower, buffer::PolarNetworkState)
    kernel! = reactive_power_kernel!(polar.device)
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    range_ = length(pv) + length(ref)
    ev = kernel!(
        buffer.qg,
        buffer.vmag, buffer.vang, buffer.pinj,
        pv, ref, pv_to_gen, ref_to_gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
        ybus_im.nzval, polar.reactive_load,
        ndrange=range_,
        dependencies=Event(polar.device)
    )
    wait(ev)
end

KA.@kernel function adj_reactive_power_edge_kernel!(
    qg, adj_qg,
    vmag, adj_vmag, vang, adj_vang,
    pinj, adj_pinj,
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

function adj_reactive_power!(
    F, adj_F, vm, adj_vm, va, adj_va,
    ybus_re, ybus_im, transpose_perm,
    pinj, adj_pinj,
    edge_vm_from, edge_vm_to, edge_va_from, edge_va_to,
    reactive_load,
    pv, pq, ref, pv_to_gen, ref_to_gen, nbus, device
)
    npv = length(pv)
    npq = length(pq)
    nvbus = length(vm)
    nnz = length(ybus_re.nzval)

    colptr = ybus_re.colptr
    rowval = ybus_re.rowval

    kernel_edge! = adj_reactive_power_edge_kernel!(device)

    range_ = length(pv) + length(ref)

    ev = kernel_edge!(
        F, adj_F,
        vm, adj_vm,
        va, adj_va,
        pinj, adj_pinj,
        pv, ref, pv_to_gen, ref_to_gen,
        edge_vm_from, edge_vm_to,
        edge_va_from, edge_va_to,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
        ybus_im.nzval, reactive_load,
        ndrange=range_,
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
            ndrange=nvbus,
            dependencies=Event(device)
        )
        wait(ev)
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

KA.@kernel function adj_branch_flow_edge_kernel!(
        adj_slines, vmag, adj_vmag, vang, adj_vang,
        adj_va_to_lines, adj_va_from_lines, adj_vm_to_lines, adj_vm_from_lines,
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
    adj_vm_to_lines[ℓ] += (2.0 * vmag[to_bus] * ytf_abs * vmag[fr_bus]^2
                      + 4.0 * vmag[to_bus]^3 * ytt_abs
                      + 6.0 * vmag[fr_bus] * vmag[to_bus]^2 * (yre_to * cosθ - yim_to * sinθ)
                       ) * adj_to_flow
    adj_vm_from_lines[ℓ] += (2.0 * vmag[to_bus]^2 * vmag[fr_bus] * ytf_abs
                      + 2.0 * vmag[to_bus]^3 * (yre_to * cosθ - yim_to * sinθ)
                        ) * adj_to_flow
    adj_cosθ = 2.0 * vmag[to_bus]^3 * vmag[fr_bus] *   yre_to  * adj_to_flow
    adj_sinθ = 2.0 * vmag[to_bus]^3 * vmag[fr_bus] * (-yim_to) * adj_to_flow

    adj_from_flow = adj_slines[ℓ]
    adj_vm_from_lines[ℓ] += (4.0 * yff_abs * vmag[fr_bus]^3
                      + 2.0 * vmag[to_bus]^2 * vmag[fr_bus] * yft_abs
                      + 6.0 * vmag[fr_bus]^2 * vmag[to_bus] * (yre_fr * cosθ - yim_fr * sinθ)
                       ) * adj_from_flow
    adj_vm_to_lines[ℓ] += (2.0 * yft_abs * vmag[fr_bus]^2 * vmag[to_bus]
                       + 2.0 * vmag[fr_bus]^3 * (yre_fr * cosθ - yim_fr * sinθ)
                        ) * adj_from_flow
    adj_cosθ += 2.0 * vmag[to_bus] * vmag[fr_bus]^3 *   yre_fr  * adj_from_flow
    adj_sinθ += 2.0 * vmag[to_bus] * vmag[fr_bus]^3 * (-yim_fr) * adj_from_flow

    adj_Δθ =   cosθ * adj_sinθ
    adj_Δθ -=  sinθ * adj_cosθ
    adj_va_from_lines[ℓ] += adj_Δθ
    adj_va_to_lines[ℓ] -= adj_Δθ
end

KA.@kernel function adj_branch_flow_node_kernel!(vm, adj_vm, va, adj_va,
        adj_va_to_lines, adj_va_from_lines, adj_vm_to_lines, adj_vm_from_lines,
        f, t, nlines
)
    i = @index(Global, Linear)
    @inbounds for ℓ in 1:nlines
        if f[ℓ] == i
            adj_vm[i] += adj_vm_from_lines[ℓ]
            adj_va[i] += adj_va_from_lines[ℓ]
        end
        if t[ℓ] == i
            adj_vm[i] += adj_vm_to_lines[ℓ]
            adj_va[i] += adj_va_to_lines[ℓ]
        end
    end
end

function adj_branch_flow!(
        adj_slines, vm, adj_vm, va, adj_va,
        adj_vm_from_lines, adj_va_from_lines, adj_vm_to_lines, adj_va_to_lines,
        yff_re, yft_re, ytf_re, ytt_re,
        yff_im, yft_im, ytf_im, ytt_im,
        f, t, nlines, device
    )
    nvbus = length(va)
    kernel_edge! = adj_branch_flow_edge_kernel!(device)
    kernel_node! = adj_branch_flow_node_kernel!(device)

    ev = kernel_edge!(
            adj_slines, vm, adj_vm, va, adj_va,
            adj_va_to_lines, adj_va_from_lines, adj_vm_to_lines, adj_vm_from_lines,
            yff_re, yft_re, ytf_re, ytt_re,
            yff_im, yft_im, ytf_im, ytt_im,
            f, t, nlines, ndrange = nlines,
            dependencies=Event(device)
    )
    wait(ev)
    ev = kernel_node!(
            vm, adj_vm, va, adj_va,
            adj_va_to_lines, adj_va_from_lines, adj_vm_to_lines, adj_vm_from_lines,
            f, t, nlines, ndrange = nvbus,
            dependencies=Event(device)
    )
    wait(ev)
end

