
is_constraint(::typeof(reactive_power_constraints)) = true

# Here, the power constraints are ordered as:
# g = [qg_gen]
function _reactive_power_constraints(
    qg, v_m, v_a, pinj, qinj, qload,
    ybus_re, ybus_im, pv, pq, ref, pv_to_gen, ref_to_gen, nbus, device
)
    kernel! = reactive_power_kernel!(device)
    range_ = length(pv) + length(ref)
    ev = kernel!(
        qg,
        v_m, v_a, pinj,
        pv, ref, pv_to_gen, ref_to_gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
        ybus_im.nzval, qload,
        ndrange=range_,
        dependencies=Event(device)
    )
    wait(ev)
end

function reactive_power_constraints(polar::PolarForm, cons, buffer)
    # Refresh reactive power generation in buffer
    update!(polar, PS.Generators(), PS.ReactivePower(), buffer)
    # Constraint on Q_ref (generator) (Q_inj = Q_g - Q_load)
    copy!(cons, buffer.qg)
    return
end

# Function for AD with ForwardDiff
function reactive_power_constraints(polar::PolarForm, cons, vm, va, pbus, qbus)
    nbus = length(vm)
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())
    _reactive_power_constraints(
        cons, vm, va, pbus, qbus, polar.reactive_load,
        ybus_re, ybus_im, pv, pq, ref, pv_to_gen, ref_to_gen, nbus, polar.device
    )
end

function size_constraint(polar::PolarForm, ::typeof(reactive_power_constraints))
    return PS.get(polar.network, PS.NumberOfGenerators())
end

function bounds(polar::PolarForm{T, IT, VT, MT}, ::typeof(reactive_power_constraints)) where {T, IT, VT, MT}
    q_min, q_max = PS.bounds(polar.network, PS.Generators(), PS.ReactivePower())
    return convert(VT, q_min), convert(VT, q_max)
end

# Adjoint
function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    cons, ∂cons,
    vm, ∂vm,
    va, ∂va,
    pinj, ∂pinj,
) where {F<:typeof(reactive_power_constraints), S, I}
    nbus = get(polar, PS.NumberOfBuses())
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    fill!(pbm.intermediate.∂edge_vm_fr , 0.0)
    fill!(pbm.intermediate.∂edge_vm_to , 0.0)
    fill!(pbm.intermediate.∂edge_va_fr , 0.0)
    fill!(pbm.intermediate.∂edge_va_to , 0.0)

    adj_reactive_power!(
        cons, ∂cons,
        vm, ∂vm,
        va, ∂va,
        ybus_re, ybus_im, polar.topology.sortperm,
        pinj, ∂pinj,
        pbm.intermediate.∂edge_vm_fr,
        pbm.intermediate.∂edge_vm_to,
        pbm.intermediate.∂edge_va_fr,
        pbm.intermediate.∂edge_va_to,
        polar.reactive_load,
        pv, pq, ref, pv_to_gen, ref_to_gen, nbus,
        polar.device
    )
end

function matpower_jacobian(polar::PolarForm, X::Union{State,Control}, ::typeof(reactive_power_constraints), V)
    nbus = get(polar, PS.NumberOfBuses())
    pf = polar.network
    ref = pf.ref
    pv = pf.pv
    pq = pf.pq
    gen2bus = polar.indexing.index_generators
    Ybus = pf.Ybus

    dSbus_dVm, dSbus_dVa = PS.matpower_residual_jacobian(V, Ybus)

    if isa(X, State)
        j11 = imag(dSbus_dVa[gen2bus, [pv; pq]])
        j12 = imag(dSbus_dVm[gen2bus, pq])
        return [j11 j12]
    elseif isa(X, Control)
        j11 = imag(dSbus_dVm[gen2bus, [ref; pv]])
        j12 = spzeros(length(gen2bus), length(pv))
        return [j11 j12]
    end
end

function matpower_hessian(polar::PolarForm, ::typeof(reactive_power_constraints), buffer, λ)
    nbus = get(polar, PS.NumberOfBuses())
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    gen2bus = polar.indexing.index_generators
    # Check consistency
    @assert length(λ) == length(gen2bus)

    λq = zeros(nbus)
    # Select only buses with generators
    λq[gen2bus] .= λ

    V = buffer.vmag .* exp.(im .* buffer.vang) |> Array
    hxx, hxu, huu = PS.reactive_power_hessian(V, polar.network.Ybus, λq, pv, pq, ref)
    return FullSpaceHessian(
        hxx, hxu, huu,
    )
end

