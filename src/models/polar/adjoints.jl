
function put!(
    polar::PolarForm{T, IT, VT, AT},
    ::State,
    ∂x::VT,
    ∂vmag::VT,
    ∂vang::VT,
) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    # build vector x
    for (i, p) in enumerate(polar.network.pv)
        ∂x[i] = ∂vang[p]
    end
    for (i, p) in enumerate(polar.network.pq)
        ∂x[npv+i] = ∂vang[p]
        ∂x[npv+npq+i] = ∂vmag[p]
    end
end

function put!(
    polar::PolarForm{T, IT, VT, AT},
    ::Control,
    ∂u::VT,
    ∂vmag::VT,
    ∂pbus::VT,
) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    # build vector u
    for (i, p) in enumerate(polar.network.ref)
        ∂u[i] = ∂vmag[p]
    end
    for (i, p) in enumerate(polar.network.pv)
        ∂u[nref + i] = 0.0
        ∂u[nref + npv + i] = ∂vmag[p]
    end
end

function put(
    polar::PolarForm{T, VT, AT},
    ::PS.Generator,
    ::PS.ActivePower,
    obj_ad::AD.ObjectiveAD,
    buffer::PolarNetworkState
) where {T, VT, AT}
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())

    index_gen = PS.get(polar.network, PS.GeneratorIndexes())
    index_pv = polar.network.pv
    index_ref = polar.network.ref
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen

    # Get voltages. This is only needed to get the size of adjvmag and adjvang
    vmag = buffer.vmag
    vang = buffer.vang

    adj_pg = obj_ad.∂pg
    adj_x = obj_ad.∇fₓ
    adj_u = obj_ad.∇fᵤ
    adj_vmag = obj_ad.∂vm
    adj_vang = obj_ad.∂va
    fill!(adj_vmag, 0.0)
    fill!(adj_vang, 0.0)
    fill!(adj_x, 0.0)
    fill!(adj_u, 0.0)

    for i in 1:nref
        bus = index_ref[i]
        i_gen = ref_to_gen[i]
        # pg[i] = inj + polar.active_load[bus]
        adj_inj = adj_pg[i_gen]
        PS.put_power_injection!(bus, vmag, vang, adj_vmag, adj_vang, adj_inj, polar.ybus_re, polar.ybus_im)
    end
    put!(polar, Control(), adj_u, adj_vmag, adj_vang)
    put!(polar, State(), adj_x, adj_vmag, adj_vang)
    for i in 1:npv
        bus = index_pv[i]
        i_gen = pv_to_gen[i]
        # pg[i] = u[nref + ipv]
        adj_u[nref + i] += adj_pg[i_gen]
    end

    return
end

function cost_production_adjoint(polar::PolarForm, ∂obj::AD.ObjectiveAD, buffer::PolarNetworkState)
    pg = buffer.pg
    coefs = polar.costs_coefficients
    # Return adjoint of quadratic cost
    @inbounds for i in eachindex(pg)
        ∂obj.∂pg[i] = coefs[i, 3] + 2.0 * coefs[i, 4] * pg[i]
    end
    put(polar, PS.Generator(), PS.ActivePower(), ∂obj, buffer)
end

