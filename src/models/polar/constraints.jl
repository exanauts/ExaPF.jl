
# By default, generic Julia functions are not considered as constraint:
is_constraint(::Function) = false

# Generic inequality constraints
# We add constraint only on vmag_pq
function state_constraint(polar::PolarForm, g, buffer)
    index_pq = polar.indexing.index_pq
    g .= @view buffer.vmag[index_pq]
    return
end
is_constraint(::typeof(state_constraint)) = true
size_constraint(polar::PolarForm{T, IT, VT, AT}, ::typeof(state_constraint)) where {T, IT, VT, AT} = PS.get(polar.network, PS.NumberOfPQBuses())
function bounds(polar::PolarForm, ::typeof(state_constraint))
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    fr_ = npq + npv + 1
    to_ = 2*npq + npv
    return polar.x_min[fr_:to_], polar.x_max[fr_:to_]
end
# State Jacobian: Jx_i = [0, ..., 1, ... 0] where
function jacobian(polar::PolarForm, ::typeof(state_constraint), i_cons, ∂jac, buffer)
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    fr_ = npq + npv

    # Adjoint / State
    fill!(∂jac.∇fₓ, 0)
    ∂jac.∇fₓ[fr_ + i_cons] = 1.0
    # Adjoint / Control
    fill!(∂jac.∇fᵤ, 0)
end
function jtprod(polar::PolarForm, ::typeof(state_constraint), ∂jac, buffer, v)
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    fr_ = npq + npv + 1
    # Adjoint / Control
    fill!(∂jac.∇fᵤ, 0)
    # Adjoint / State
    fill!(∂jac.∇fₓ, 0)
    ∂jac.∇fₓ[fr_:end] .= v
end

# Here, the power constraints are ordered as:
# g = [P_ref; Q_ref; Q_pv]
function power_constraints(polar::PolarForm, g, buffer)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    Vm, Va, pbus, qbus = buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj

    cnt = 1
    # Constraint on P_ref (generator) (P_inj = P_g - P_load)
    for bus in ref
        g[cnt] = get_power_injection(bus, Vm, Va, polar.ybus_re, polar.ybus_im) + polar.active_load[bus]
        cnt += 1
    end
    # Constraint on Q_ref (generator) (Q_inj = Q_g - Q_load)
    for bus in ref
        g[cnt] = get_react_injection(bus, Vm, Va, polar.ybus_re, polar.ybus_im) + polar.reactive_load[bus]
        cnt += 1
    end
    # Constraint on Q_pv (generator) (Q_inj = Q_g - Q_load)
    for bus in pv
        g[cnt] = get_react_injection(bus, Vm, Va, polar.ybus_re, polar.ybus_im) + polar.reactive_load[bus]
        cnt += 1
    end
    return
end
is_constraint(::typeof(power_constraints)) = true
function size_constraint(polar::PolarForm{T, IT, VT, AT}, ::typeof(power_constraints)) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    return 2*nref + npv
end
function bounds(polar::PolarForm{T, IT, VT, AT}, ::typeof(power_constraints)) where {T, IT, VT, AT}
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())

    # Get all bounds (lengths of p_min, p_max, q_min, q_max equal to ngen)
    p_min, p_max = PS.bounds(polar.network, PS.Generator(), PS.ActivePower())
    q_min, q_max = PS.bounds(polar.network, PS.Generator(), PS.ReactivePower())

    index_ref = polar.indexing.index_ref
    index_pv = polar.indexing.index_pv
    index_gen = polar.indexing.index_generators
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen

    # Remind that the ordering is
    # g = [P_ref; Q_ref; Q_pv]
    MT = polar.AT
    pq_min = [p_min[ref_to_gen]; q_min[ref_to_gen]; q_min[pv_to_gen]]
    pq_max = [p_max[ref_to_gen]; q_max[ref_to_gen]; q_max[pv_to_gen]]
    return convert(MT, pq_min), convert(MT, pq_max)
end
# Jacobian
function jacobian(
    polar::PolarForm,
    ::typeof(power_constraints),
    i_cons,
    ∂jac,
    buffer
)
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    index_pv = polar.network.pv
    index_ref = polar.network.ref

    vmag = buffer.vmag
    vang = buffer.vang
    adj_x = ∂jac.∇fₓ
    adj_u = ∂jac.∇fᵤ
    adj_vmag = ∂jac.∂vm
    adj_vang = ∂jac.∂va
    fill!(adj_vmag, 0.0)
    fill!(adj_vang, 0.0)
    fill!(adj_x, 0.0)
    fill!(adj_u, 0.0)

    if i_cons <= nref
        # Constraint on P_ref (generator) (P_inj = P_g - P_load)
        bus = index_ref[i_cons]
        adj_inj = 1.0
        put_active_power_injection!(bus, vmag, vang, adj_vmag, adj_vang, adj_inj, polar.ybus_re, polar.ybus_im)
    elseif i_cons <= 2*nref
        # Constraint on Q_ref (generator) (Q_inj = Q_g - Q_load)
        bus = index_ref[i_cons - nref]
        adj_inj = 1.0
        put_reactive_power_injection!(bus, vmag, vang, adj_vmag, adj_vang, adj_inj, polar.ybus_re, polar.ybus_im)
    else
        # Constraint on Q_pv (generator) (Q_inj = Q_g - Q_load)
        adj_inj = 1.0
        bus = index_pv[i_cons - 2* nref]
        put_reactive_power_injection!(bus, vmag, vang, adj_vmag, adj_vang, adj_inj, polar.ybus_re, polar.ybus_im)
    end
    put!(polar, Control(), adj_u, adj_vmag, adj_vang)
    put!(polar, State(), adj_x, adj_vmag, adj_vang)
end
# Jacobian-transpose vector product
function jtprod(
    polar::PolarForm,
    ::typeof(power_constraints),
    ∂jac,
    buffer,
    v::AbstractVector,
)
    m = size_constraint(polar, power_constraints)
    jvx = similar(∂jac.∇fₓ) ; fill!(jvx, 0)
    jvu = similar(∂jac.∇fᵤ) ; fill!(jvu, 0)
    for i_cons in 1:m
        if !iszero(v[i_cons])
            jacobian(polar, power_constraints, i_cons, ∂jac, buffer)
            jx, ju = ∂jac.∇fₓ, ∂jac.∇fᵤ
            jvx .+= jx .* v[i_cons]
            jvu .+= ju .* v[i_cons]
        end
    end
    ∂jac.∇fₓ .= jvx
    ∂jac.∇fᵤ .= jvu
end

