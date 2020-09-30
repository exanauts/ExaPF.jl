
function get(
    polar::PolarForm{T, IT, VT, AT},
    ::State,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
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
function get!(
    polar::PolarForm{T, IT, VT, AT},
    ::State,
    x::AbstractVector,
    buffer::PolarNetworkState
) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    # Copy values of vang and vmag into x
    # NB: this leads to 3 memory allocation on the GPU
    #     we use indexing on the CPU, as for some reason
    #     we get better performance than with the indexing on the GPU
    #     stored in the buffer polar.indexing.
    x[1:npv] .= @view buffer.vang[polar.network.pv]
    x[npv+1:npv+npq] .= @view buffer.vang[polar.network.pq]
    x[npv+npq+1:npv+2*npq] .= @view buffer.vmag[polar.network.pq]
end

function get(
    polar::PolarForm{T, IT, VT, AT},
    ::Control,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
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
    u[nref + 1:nref + npv] = pbus[polar.network.pv] + pload[polar.network.pv]
    u[nref + npv + 1:nref + 2*npv] = vmag[polar.network.pv]
    return u
end

function get(
    polar::PolarForm{T, IT, VT, AT},
    ::Parameters,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    # build vector p
    dimension = nref + 2*npq
    p = VT(undef, dimension)
    p[1:nref] = vang[polar.network.ref]
    p[nref + 1:nref + npq] = pbus[polar.network.pq]
    p[nref + npq + 1:nref + 2*npq] = qbus[polar.network.pq]
    return p
end

# Bridge with buses' attributes
function get(polar::PolarForm{T, IT, VT, AT}, ::PS.Buses, ::PS.VoltageMagnitude, x, u, p; V=eltype(x)) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    MT = polar.AT
    vmag = MT{V, 1}(undef, nbus)
    vmag[polar.network.pq] = x[npq+npv+1:end]
    vmag[polar.network.ref] = u[1:nref]
    vmag[polar.network.pv] = u[nref + npv + 1:nref + 2*npv]
    return vmag
end
function get(polar::PolarForm{T, IT, VT, AT}, ::PS.Buses, ::PS.VoltageAngle, x, u, p; V=eltype(x)) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    MT = polar.AT
    vang = MT{V, 1}(undef, nbus)
    vang[polar.network.pq] = x[npv+1:npv+npq]
    vang[polar.network.pv] = x[1:npv]
    vang[polar.network.ref] = p[1:nref]
    return vang
end
function get(polar::PolarForm{T, IT, VT, AT}, ::PS.Buses, ::PS.ActivePower, x, u, p; V=eltype(x)) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    vmag = get(polar, PS.Buses(), PS.VoltageMagnitude(), x, u, p)
    vang = get(polar, PS.Buses(), PS.VoltageAngle(), x, u, p)
    MT = polar.AT
    pinj = MT{V, 1}(undef, nbus)
    pinj[polar.network.pv] = u[nref + 1:nref + npv] - polar.active_load[polar.network.pv]
    pinj[polar.network.pq] = p[nref + 1:nref + npq]
    for bus in polar.network.ref
        pinj[bus] = PS.get_power_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end
    return pinj
end
function get(polar::PolarForm{T, IT, VT, AT}, ::PS.Buses, ::PS.ReactivePower, x, u, p; V=eltype(x)) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    vmag = get(polar, PS.Buses(), PS.VoltageMagnitude(), x, u, p)
    vang = get(polar, PS.Buses(), PS.VoltageAngle(), x, u, p)
    qinj = VT(undef, nbus)
    qinj[polar.network.pq] = p[nref + npq + 1:nref + 2*npq]
    for bus in [polar.network.ref; polar.network.pv]
        qinj[bus] = PS.get_react_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end
    return qinj
end

# Bridge with generators' attributes
function get(polar::PolarForm{T, IT, VT, AT}, ::PS.Generator, ::PS.ActivePower, x, u, p; V=eltype(x)) where {T, IT, VT, AT}
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())

    index_ref = polar.indexing.index_ref
    index_pv = polar.indexing.index_pv
    index_gen = polar.indexing.index_generators

    # Get voltages.
    vmag = get(polar, PS.Buses(), PS.VoltageMagnitude(), x, u, p; V=V)
    vang = get(polar, PS.Buses(), PS.VoltageAngle(), x, u, p; V=V)

    MT = polar.AT
    pg = MT{V, 1}(undef, ngen)
    # TODO: check the complexity of this for loop
    for i in 1:ngen
        bus = index_gen[i]
        if bus in index_ref
            inj = PS.get_power_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
            pg[i] = inj + polar.active_load[bus]
        else
            ipv = findfirst(isequal(bus), index_pv)
            pg[i] = u[nref + ipv]
        end
    end

    return pg
end

function get_network_state(polar::PolarForm{T, IT, VT, AT}, x, u, p; V=Float64) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    pf = polar.network

    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq

    MT = polar.AT
    vmag = MT{V, 1}(undef, nbus)
    vang = MT{V, 1}(undef, nbus)
    pinj = MT{V, 1}(undef, nbus)
    qinj = MT{V, 1}(undef, nbus)

    vang[pv] .= x[1:npv]
    vang[pq] .= x[npv+1:npv+npq]
    vmag[pq] .= x[npv+npq+1:end]

    vmag[ref] .= u[1:nref]
    pinj[pv] .= u[nref + 1:nref + npv] - polar.active_load[pv]
    vmag[pv] .= u[nref + npv + 1:nref + 2*npv]

    vang[ref] .= p[1:nref]
    pinj[pq] .= p[nref + 1:nref + npq]
    qinj[pq] .= p[nref + npq + 1:nref + 2*npq]

    for bus in ref
        pinj[bus] = PS.get_power_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
        qinj[bus] = PS.get_react_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end

    for bus in pv
        qinj[bus] = PS.get_react_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end

    return vmag, vang, pinj, qinj
end
