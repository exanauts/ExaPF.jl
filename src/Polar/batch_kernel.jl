KA.@kernel function batch_adj_residual_edge_kernel!(
    adj_F, vm, adj_vm, va, adj_va,
    colptr, rowval,
    ybus_re_nzval, ybus_im_nzval,
    edge_vm_from, edge_vm_to,
    edge_va_from, edge_va_to,
    pinj, adj_pinj, qinj, pv, pq
)
    i, j = @index(Global, NTuple)

    npv = size(pv, 1)
    npq = size(pq, 1)

    # REAL PV: 1:npv
    # REAL PQ: (npv+1:npv+npq)
    # IMAG PQ: (npv+npq+1:npv+2npq)
    fr = (i <= npv) ? pv[i] : pq[i - npv]
    for c in colptr[fr]:colptr[fr+1]-1
        # Forward loop
        to = rowval[c]
        aij = va[fr, j] - va[to, j]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = vm[fr, j]*vm[to, j]*ybus_re_nzval[c]
        coef_sin = vm[fr, j]*vm[to, j]*ybus_im_nzval[c]

        cos_val = cos(aij)
        sin_val = sin(aij)

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

        edge_vm_from[c, j] += vm[to, j]*ybus_im_nzval[c]*adj_coef_sin
        edge_vm_to[c, j]   += vm[fr, j]*ybus_im_nzval[c]*adj_coef_sin
        edge_vm_from[c, j] += vm[to, j]*ybus_re_nzval[c]*adj_coef_cos
        edge_vm_to[c, j]   += vm[fr, j]*ybus_re_nzval[c]*adj_coef_cos

        edge_va_from[c, j] += adj_aij
        edge_va_to[c, j]   -= adj_aij
    end
    # qinj is not active
    # if i > npv
    #     adj_qinj[fr] -= adj_F[i + npq]
    # end
    # adj_pinj[j, fr] -= adj_F[j, i]
end

KA.@kernel function batch_adj_node_kernel!(
    adj_vm, adj_va,
    colptr, rowval,
    edge_vm_from, edge_vm_to,
    edge_va_from, edge_va_to, dest
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
function batch_adj_residual_polar!(
    F, adj_F, vm, adj_vm, va, adj_va,
    ybus_re, ybus_im, transpose_perm,
    pinj, adj_pinj, qinj,
    edge_vm_from, edge_vm_to, edge_va_from, edge_va_to,
    pv, pq, nbus
) where {T}
    if isa(F, CUDA.CuArray)
        device = KA.CUDADevice()
    else
        device = KA.CPU()
    end
    npv = length(pv)
    npq = length(pq)
    nvbus = length(vm)
    nnz = length(ybus_re.nzval)
    colptr = ybus_re.colptr
    rowval = ybus_re.rowval

    nbatch = size(F, 2)
    nvars = npv+npq
    ndrange = (nvars, nbatch)

    kernel_edge! = batch_adj_residual_edge_kernel!(device)
    ev = kernel_edge!(adj_F, vm, adj_vm, va, adj_va,
        ybus_re.colptr, ybus_re.rowval,
        ybus_re.nzval, ybus_im.nzval,
        edge_vm_from, edge_vm_to,
        edge_va_from, edge_va_to,
        pinj, adj_pinj, qinj, pv, pq,
        ndrange=ndrange,
        dependencies = Event(device)
    )
    wait(ev)

    # The permutation corresponding to the transpose of Ybus.
    # is given in transpose_perm
    ndrange = (nbus, nbatch)

    ev = batch_adj_node_kernel!(device)(
        adj_vm, adj_va,
        ybus_re.colptr, ybus_re.rowval,
        edge_vm_from, edge_vm_to,
        edge_va_from, edge_va_to, transpose_perm,
        ndrange=ndrange,
        dependencies = Event(device),
        workgroupsize=256
    )
    wait(ev)
end
