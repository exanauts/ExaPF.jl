
# Generic inequality constraints
# We add constraint only on vmag_pq
function state_constraint(polar::PolarForm, g, buffer)
    index_pq = polar.indexing.index_pq
    g .= @view buffer.vmag[index_pq]
    return
end
size_constraint(polar::PolarForm{T, IT, VT, AT}, ::typeof(state_constraint)) where {T, IT, VT, AT} = PS.get(polar.network, PS.NumberOfPQBuses())
function bounds(polar::PolarForm, ::typeof(state_constraint))
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    fr_ = npq + npv + 1
    to_ = 2*npq + npv
    return polar.x_min[fr_:to_], polar.x_max[fr_:to_]
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
        g[cnt] = PS.get_power_injection(bus, Vm, Va, polar.ybus_re, polar.ybus_im) + polar.active_load[bus]
        cnt += 1
    end
    # Constraint on Q_ref (generator) (Q_inj = Q_g - Q_load)
    for bus in ref
        g[cnt] = PS.get_react_injection(bus, Vm, Va, polar.ybus_re, polar.ybus_im) + polar.reactive_load[bus]
        cnt += 1
    end
    # Constraint on Q_pv (generator) (Q_inj = Q_g - Q_load)
    for bus in pv
        g[cnt] = PS.get_react_injection(bus, Vm, Va, polar.ybus_re, polar.ybus_im) + polar.reactive_load[bus]
        cnt += 1
    end
    return
end
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
