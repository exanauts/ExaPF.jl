is_constraint(::typeof(bus_power_injection)) = true
size_constraint(polar::PolarForm, ::typeof(bus_power_injection)) = 2 * get(polar, PS.NumberOfBuses())

KA.@kernel function bus_power_injection_kernel!(
    inj, @Const(vmag), @Const(vang),
    @Const(colptr), @Const(rowval),
    @Const(ybus_re_nzval), @Const(ybus_im_nzval), nbus,
)
    bus, j = @index(Global, NTuple)

    @inbounds for c in colptr[bus]:colptr[bus+1]-1
        to = rowval[c]
        aij = vang[bus, j] - vang[to, j]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = vmag[bus, j]*vmag[to, j]*ybus_re_nzval[c]
        coef_sin = vmag[bus, j]*vmag[to, j]*ybus_im_nzval[c]
        cos_val = cos(aij)
        sin_val = sin(aij)

        inj[bus, j] += coef_cos * cos_val + coef_sin * sin_val
        inj[bus+nbus, j] += coef_cos * sin_val - coef_sin * cos_val
    end
end

KA.@kernel function adj_bus_power_injection_kernel!(
    edge_vm_from, edge_vm_to,
    edge_va_from, edge_va_to,
    @Const(adj_inj), @Const(vmag), @Const(vang),
    @Const(colptr), @Const(rowval),
    @Const(ybus_re_nzval), @Const(ybus_im_nzval), nbus,
)
    bus, j = @index(Global, NTuple)

    @inbounds for c in colptr[bus]:colptr[bus+1]-1
        # Forward loop
        to = rowval[c]
        aij = vang[bus, j] - vang[to, j]
        v_fr = vmag[bus, j]
        v_to = vmag[to,  j]
        y_re = ybus_re_nzval[c]
        y_im = ybus_im_nzval[c]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = v_fr*v_to*y_re[c]
        coef_sin = v_fr*v_to*y_im[c]

        cos_val = cos(aij)
        sin_val = sin(aij)

        adj_coef_cos = cos_val  * adj_inj[bus]
        adj_coef_sin = sin_val  * adj_inj[bus]
        adj_cos_val  = coef_cos * adj_inj[bus]
        adj_sin_val  = coef_sin * adj_inj[bus]

        adj_coef_cos +=  sin_val  * adj_inj[bus+nbus]
        adj_coef_sin += -cos_val  * adj_inj[bus+nbus]
        adj_cos_val  += -coef_sin * adj_inj[bus+nbus]
        adj_sin_val  +=  coef_cos * adj_inj[bus+nbus]

        adj_aij =   cos_val * adj_sin_val
        adj_aij += -sin_val * adj_cos_val

        edge_vm_from[c, j] += v_to * y_im * adj_coef_sin
        edge_vm_to[c, j]   += v_fr * y_im * adj_coef_sin
        edge_vm_from[c, j] += v_to * y_re * adj_coef_cos
        edge_vm_to[c, j]   += v_fr * y_re * adj_coef_cos

        edge_va_from[c, j] += adj_aij
        edge_va_to[c, j]   -= adj_aij
    end
end

function bus_power_injection(polar::PolarForm, cons, vmag, vang, pnet, qnet, pload, qload)
    nbus = get(polar, PS.NumberOfBuses())
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())
    fill!(cons, 0)
    ndrange = (nbus, size(cons, 2))
    ev = bus_power_injection_kernel!(polar.device)(
        cons, vmag, vang,
        ybus_re.colptr, ybus_re.rowval, ybus_re.nzval, ybus_im.nzval, nbus,
        ndrange=ndrange, dependencies=Event(polar.device),
    )
    wait(ev)
end

function bus_power_injection(polar::PolarForm, cons, buffer::PolarNetworkState)
    bus_power_injection(
        polar, cons,
        buffer.vmag, buffer.vang,
        buffer.pnet, buffer.qnet,
        buffer.pload, buffer.qload,
    )
end

# Adjoint with standardized interface
function _adjoint_bus_power_injection!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory,
    ∂cons, vmag, ∂vmag, vang, ∂vang,
)
    nbus = get(polar, PS.NumberOfBuses())
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    fill!(pbm.intermediate.∂edge_vm_fr , 0.0)
    fill!(pbm.intermediate.∂edge_vm_to , 0.0)
    fill!(pbm.intermediate.∂edge_va_fr , 0.0)
    fill!(pbm.intermediate.∂edge_va_to , 0.0)

    ndrange = (nbus, size(vmag, 2))
    # ADJOINT WRT EDGES
    adj_bus_power_injection_kernel!(polar.device)(
        pbm.intermediate.∂edge_vm_fr,
        pbm.intermediate.∂edge_vm_to,
        pbm.intermediate.∂edge_va_fr,
        pbm.intermediate.∂edge_va_to,
        ∂cons,
        vmag, vang,
        ybus_re.colptr, ybus_re.rowval, ybus_re.nzval, ybus_im.nzval, nbus,
        ndrange=ndrange, dependencies=Event(polar.device),
    )

    # ADJOINT WRT NODES
    ev = gpu_adj_node_kernel!(polar.device)(
        ∂vmag, ∂vang,
        ybus_re.colptr, ybus_re.rowval,
        pbm.intermediate.∂edge_vm_fr, pbm.intermediate.∂edge_vm_to,
        pbm.intermediate.∂edge_va_fr, pbm.intermediate.∂edge_va_to,
        polar.topology.sortperm,
        ndrange=ndrange, dependencies=Event(polar.device)
    )
    wait(ev)
end

function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    cons, ∂cons,
    vm, ∂vm,
    va, ∂va,
    pnet, ∂pnet,
    pload, qload,
) where {F<:typeof(bus_power_injection), S, I}
    fill!(∂vm, 0)
    fill!(∂va, 0)
    _adjoint_bus_power_injection!(polar, pbm, ∂cons, vm, ∂vm, va, ∂va)
    return
end

