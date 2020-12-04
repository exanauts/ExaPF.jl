
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
    x = xzeros(VT, dimension)
    x[1:npv] = vang[polar.network.pv]
    x[npv+1:npv+npq] = vang[polar.network.pq]
    x[npv+npq+1:end] = vmag[polar.network.pq]

    return x
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
    u = xzeros(VT, dimension)
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
    p = xzeros(VT, dimension)
    p[1:nref] = vang[polar.network.ref]
    p[nref + 1:nref + npq] = pbus[polar.network.pq]
    p[nref + npq + 1:nref + 2*npq] = qbus[polar.network.pq]
    return p
end

