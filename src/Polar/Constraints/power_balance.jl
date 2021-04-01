is_constraint(::typeof(power_balance)) = true

function _power_balance!(
    F, v_m, v_a, pinj, qinj, ybus_re, ybus_im, pv, pq, ref, nbus,
)
    npv = length(pv)
    npq = length(pq)
    if isa(F, Array)
        device = KA.CPU()
        kernel! = residual_kernel!(device)
    else
        device = KA.CUDADevice()
        kernel! = residual_kernel!(device)
    end
    ev = kernel!(
        F, v_m, v_a,
        ybus_re.colptr, ybus_re.rowval,
        ybus_re.nzval, ybus_im.nzval,
        pinj, qinj, pv, pq, nbus,
        ndrange=npv+npq,
        dependencies=Event(device)
    )
    wait(ev)
end

function power_balance(polar::PolarForm, cons, vm, va, pbus, qbus)
    nbus = get(polar, PS.NumberOfBuses())
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    fill!(cons, 0.0)
    _power_balance!(
        cons, vm, va, pbus, qbus,
        ybus_re, ybus_im,
        pv, pq, ref, nbus
    )
end

function power_balance(polar::PolarForm, cons, buffer::PolarNetworkState)
    Vm, Va, pbus, qbus = buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj
    power_balance(polar, cons, Vm, Va, pbus, qbus)
end

function size_constraint(polar::PolarForm, ::typeof(power_balance))
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    return 2 * npq + npv
end

function bounds(polar::PolarForm{T, IT, VT, MT}, ::typeof(power_balance)) where {T, IT, VT, MT}
    m = size_constraint(polar, power_balance)
    return xzeros(VT, m) , xzeros(VT, m)
end

# Adjoint
function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    cons, ∂cons,
    vm, ∂vm,
    va, ∂va,
    pinj, ∂pinj,
) where {F<:typeof(power_balance), S, I}
    nbus = get(polar, PS.NumberOfBuses())
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    qinj = polar.reactive_load
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    fill!(pbm.intermediate.∂edge_vm_fr , 0.0)
    fill!(pbm.intermediate.∂edge_vm_to , 0.0)
    fill!(pbm.intermediate.∂edge_va_fr , 0.0)
    fill!(pbm.intermediate.∂edge_va_to , 0.0)

    adj_residual_polar!(
        cons, ∂cons,
        vm, ∂vm,
        va, ∂va,
        ybus_re, ybus_im, polar.topology.sortperm,
        pinj, ∂pinj, qinj,
        pbm.intermediate.∂edge_vm_fr,
        pbm.intermediate.∂edge_vm_to,
        pbm.intermediate.∂edge_va_fr,
        pbm.intermediate.∂edge_va_to,
        pv, pq, nbus,
    )
end

function matpower_jacobian(polar::PolarForm, X::Union{State, Control}, ::typeof(power_balance), V)
    nbus = get(polar, PS.NumberOfBuses())
    pf = polar.network
    ref = pf.ref ; nref = length(ref)
    pv = pf.pv ; npv = length(pv)
    pq = pf.pq ; npq = length(pq)
    Ybus = pf.Ybus

    dSbus_dVm, dSbus_dVa = PS.matpower_residual_jacobian(V, Ybus)

    if isa(X, State)
        j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
        j12 = real(dSbus_dVm[[pv; pq], pq])
        j21 = imag(dSbus_dVa[pq, [pv; pq]])
        j22 = imag(dSbus_dVm[pq, pq])
        return [j11 j12; j21 j22]
    elseif isa(X, Control)
        j11 = real(dSbus_dVm[[pv; pq], [ref; pv]])
        j12 = sparse(I, npv + npq, npv)
        j21 = imag(dSbus_dVm[pq, [ref; pv]])
        j22 = spzeros(npq, npv)
        return [j11 -j12; j21 j22]
    end
end

# Hessian
function matpower_hessian(
    polar::PolarForm,
    ::typeof(power_balance),
    buffer::PolarNetworkState,
    λ::AbstractVector,
)
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    V = buffer.vmag .* exp.(im .* buffer.vang)
    hxx, hxu, huu = PS.residual_hessian(V, polar.network.Ybus, λ, pv, pq, ref)
    return FullSpaceHessian(hxx, hxu, huu)
end

