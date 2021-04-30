is_constraint(::typeof(power_balance)) = true

function _power_balance!(
    F, v_m, v_a, pinj, pload, qload, ybus_re, ybus_im, pv, pq, ref, nbus, device
)
    npv = length(pv)
    npq = length(pq)
    kernel! = residual_kernel!(device)
    ndrange = (npv+npq, size(F, 2))
    ev = kernel!(
        F, v_m, v_a,
        ybus_re.colptr, ybus_re.rowval,
        ybus_re.nzval, ybus_im.nzval,
        pinj, pload, qload, pv, pq, nbus,
        ndrange=ndrange,
        dependencies=Event(device)
    )
    wait(ev)
end

function power_balance(polar::PolarForm, cons, vm, va, pbus, qbus, pd, qd)
    nbus = get(polar, PS.NumberOfBuses())
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    fill!(cons, 0.0)
    _power_balance!(
        cons, vm, va, pbus, pd, qd,
        ybus_re, ybus_im,
        pv, pq, ref, nbus, polar.device
    )
end

function power_balance(polar::PolarForm, cons, buffer::PolarNetworkState)
    power_balance(
        polar, cons,
        buffer.vmag, buffer.vang,
        buffer.pinj, buffer.qinj,
        buffer.pd, buffer.qd,
    )
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
    pload, qload,
) where {F<:typeof(power_balance), S, I}
    nbus = get(polar, PS.NumberOfBuses())
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
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
        pinj, ∂pinj, pload, qload,
        pbm.intermediate.∂edge_vm_fr,
        pbm.intermediate.∂edge_vm_to,
        pbm.intermediate.∂edge_va_fr,
        pbm.intermediate.∂edge_va_to,
        pv, pq, nbus,
        polar.device
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
    λ_host = λ |> Array
    V = buffer.vmag .* exp.(im .* buffer.vang) |> Array
    hxx, hxu, huu = PS.residual_hessian(V, polar.network.Ybus, λ_host, pv, pq, ref)
    return FullSpaceHessian(hxx, hxu, huu)
end

