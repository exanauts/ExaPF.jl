abstract type AbstractBuffer end
abstract type AbstractNetworkBuffer <: AbstractBuffer end

"Store indexing on target device"
struct IndexingCache{IVT} <: AbstractBuffer
    index_pv::IVT
    index_pq::IVT
    index_ref::IVT
    index_generators::IVT
    index_pv_to_gen::IVT
    index_ref_to_gen::IVT
end

"""
    PolarNetworkState{VI,VT} <: AbstractNetworkBuffer

Buffer to store current values of all the variables describing
the network, in polar formulation. Attributes are:

- `vmag` (length: nbus): voltage magnitude at each bus
- `vang` (length: nbus): voltage angle at each bus
- `pinj` (length: nbus): power injection RHS. Equal to `Cg * Pg - Pd`
- `qinj` (length: nbus): power injection RHS. Equal to `Cg * Qg - Qd`
- `pg`   (length: ngen): active power of generators
- `qg`   (length: ngen): reactive power of generators
- `dx`   (length: nstates): cache the difference between two consecutive states (used in power flow resolution)
- `balance` (length: nstates): cache for current power imbalance (used in power flow resolution)
- `bus_gen` (length: ngen): generator-bus incidence matrix `Cg`

"""
struct PolarNetworkState{VI,VT} <: AbstractNetworkBuffer
    vmag::VT
    vang::VT
    pinj::VT
    qinj::VT
    pg::VT
    qg::VT
    balance::VT
    dx::VT
    bus_gen::VI   # Generator-Bus incidence matrix
end

function PolarNetworkState{VT}(nbus::Int, ngen::Int, nstates::Int, bus_gen::VI) where {VI, VT}
    # Bus variables
    pbus = xzeros(VT, nbus)
    qbus = xzeros(VT, nbus)
    vmag = xzeros(VT, nbus)
    vang = xzeros(VT, nbus)
    # Generators variables
    pg = xzeros(VT, ngen)
    qg = xzeros(VT, ngen)
    # Buffers
    balance = xzeros(VT, nstates)
    dx = xzeros(VT, nstates)
    return PolarNetworkState{VI,VT}(vmag, vang, pbus, qbus, pg, qg, balance, dx, bus_gen)
end

setvalues!(buf::PolarNetworkState, ::PS.VoltageMagnitude, values) = copyto!(buf.vmag, values)
setvalues!(buf::PolarNetworkState, ::PS.VoltageAngle, values) = copyto!(buf.vang, values)
function setvalues!(buf::PolarNetworkState, ::PS.ActivePower, values)
    pgenbus = view(buf.pinj, buf.bus_gen)
    # Remove previous values
    pgenbus .-= buf.pg
    # Add new values
    copyto!(buf.pg, values)
    pgenbus .+= buf.pg
end
function setvalues!(buf::PolarNetworkState, ::PS.ReactivePower, values)
    qgenbus = view(buf.qinj, buf.bus_gen)
    # Remove previous values
    qgenbus .-= buf.qg
    # Add new values
    copyto!(buf.qg, values)
    qgenbus .+= buf.qg
end
function setvalues!(buf::PolarNetworkState, ::PS.ActiveLoad, values)
    fill!(buf.pinj, 0)
    # Pbus = Cg * Pg - Pd
    buf.pinj .-= values
    pgenbus = view(buf.pinj, buf.bus_gen)
    pgenbus .+= buf.pg
end
function setvalues!(buf::PolarNetworkState, ::PS.ReactiveLoad, values)
    fill!(buf.qinj, 0)
    # Qbus = Cg * Qg - Qd
    buf.qinj .-= values
    qgenbus = view(buf.qinj, buf.bus_gen)
    qgenbus .+= buf.qg
end

function Base.iszero(buf::PolarNetworkState)
    return iszero(buf.pinj) &&
        iszero(buf.qinj) &&
        iszero(buf.vmag) &&
        iszero(buf.vang) &&
        iszero(buf.pg) &&
        iszero(buf.qg) &&
        iszero(buf.balance) &&
        iszero(buf.dx)
end

