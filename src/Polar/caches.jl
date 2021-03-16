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
    PolarNetworkState{VI, VT} <: AbstractNetworkBuffer

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

"Store topology of the network on target device."
struct NetworkTopology{VTI, VTD}
    # Bus admittance matrix
    ybus_re::Spmat{VTI, VTD} # nb x nb
    ybus_im::Spmat{VTI, VTD} # nb x nb
    # Branches admittance matrix
    ## Real part
    yff_re::VTD # nl
    yft_re::VTD # nl
    ytf_re::VTD # nl
    ytt_re::VTD # nl
    ## Imag part
    yff_im::VTD # nl
    yft_im::VTD # nl
    ytf_im::VTD # nl
    ytt_im::VTD # nl
    # Correspondence
    f_buses::VTI # nl
    t_buses::VTI # nl
    sortperm::VTI # nnz
end

function NetworkTopology{VTI, VTD}(pf::PS.PowerNetwork) where {VTI, VTD}
    Y = pf.Ybus
    ybus_re, ybus_im = Spmat{VTI, VTD}(Y)
    lines = pf.lines
    yff_re = real.(lines.Yff) |> VTD
    yft_re = real.(lines.Yft) |> VTD
    ytf_re = real.(lines.Ytf) |> VTD
    ytt_re = real.(lines.Ytt) |> VTD

    yff_im = imag.(lines.Yff) |> VTD
    yft_im = imag.(lines.Yft) |> VTD
    ytf_im = imag.(lines.Ytf) |> VTD
    ytt_im = imag.(lines.Ytt) |> VTD

    f = lines.from_buses |> VTI
    t = lines.to_buses   |> VTI
    i, j, _ = findnz(Y)
    sp = sortperm(i) |> VTI

    return NetworkTopology(
        ybus_re, ybus_im,
        yff_re, yft_re, ytf_re, ytt_re,
        yff_im, yft_im, ytf_im, ytt_im,
        f, t, sp,
    )
end

get(net::NetworkTopology, ::PS.BusAdmittanceMatrix) = (net.ybus_re, net.ybus_im)

struct HessianStructure{IT} <: AbstractStructure where {IT}
    map::IT
end

