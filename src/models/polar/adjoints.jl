
function put(
    polar::PolarForm{T, IT, VT, AT},
    ::State,
    vmag::VT,
    vang::VT,
) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    # build vector x
    dimension = get(polar, NumberOfState())
    x = VT(undef, dimension)
    x[1:npv] = vang[polar.network.pv]
    x[npv+1:npv+npq] = vang[polar.network.pq]
    x[npv+npq+1:end] = vmag[polar.network.pq]

    return x
end

function put(
    polar::PolarForm{T, IT, VT, AT},
    ::Control,
    vmag::VT,
    pbus::VT,
) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    pload = polar.active_load
    # build vector u
    dimension = get(polar, NumberOfControl())
    u = VT(undef, dimension)
    u[1:nref] = vmag[polar.network.ref]
    # u is equal to active power of generator (Pᵍ)
    # As P = Pᵍ - Pˡ , we get
    u[nref + 1:nref + npv] .= 0.0
    u[nref + npv + 1:nref + 2*npv] = vmag[polar.network.pv]
    return u
end

function put(polar::PolarForm{T, VT, AT}, ::PS.Generator, ::PS.ActivePower, cache, adj_pg) where {T, VT, AT}
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())

    index_gen = PS.get(polar.network, PS.GeneratorIndexes())
    index_pv = polar.network.pv
    index_ref = polar.network.ref

    # Get voltages. This is only needed to get the size of adjvmag and adjvang
    vmag = cache.vmag
    vang = cache.vang

    adj_vmag = similar(vmag)
    adj_vang = similar(vang)
    fill!(adj_vmag, 0.0)
    fill!(adj_vang, 0.0)

    adj_u_tmp = polar.AT{Float64, 1}(undef, get(polar, NumberOfControl()))
    fill!(adj_u_tmp, 0.0)

    for i in 1:ngen
        bus = index_gen[i]
        if bus in index_ref
            # pg[i] = inj + polar.active_load[bus]
            adj_inj = adj_pg[i]
            PS.put_power_injection!(bus, vmag, vang, adj_vmag, adj_vang, adj_inj, polar.ybus_re, polar.ybus_im)
        else
            ipv = findfirst(isequal(bus), index_pv)
            # pg[i] = u[nref + ipv]
            adj_u_tmp[nref + ipv] += adj_pg[i]
        end
    end
    adj_u = put(polar, Control(), adj_vmag, adj_vang)
    adj_x = put(polar, State(), adj_vmag, adj_vang)
    adj_u .+= adj_u_tmp

    return adj_x, adj_u
end

function cost_production_adjoint(polar::PolarForm, cache::NetworkState)
    pg = cache.pg
    c1 = polar.costs_coefficients[:, 3]
    c2 = polar.costs_coefficients[:, 4]
    # Return adjoint of quadratic cost
    adj_power_generations = c1 .+ 2.0 * c2 .* pg
    adj_x, adj_u = put(polar, PS.Generator(), PS.ActivePower(), cache, adj_power_generations)
    return adj_x, adj_u
end

