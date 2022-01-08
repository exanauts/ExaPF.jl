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
index_buses(idx::IndexingCache) = (idx.index_ref, idx.index_pv, idx.index_pq)
index_generators(idx::IndexingCache) = (idx.index_generators, idx.index_ref_to_gen, idx.index_pv_to_gen)

"""
    PolarNetworkState{VI, VT} <: AbstractNetworkBuffer

Buffer to store current values of all the variables describing
the network, in polar formulation. Attributes are:

- `vmag` (length: nbus): voltage magnitude at each bus
- `vang` (length: nbus): voltage angle at each bus
- `pnet` (length: nbus): power generation RHS. Equal to `Cg * Pg`
- `qnet` (length: nbus): power generation RHS. Equal to `Cg * Qg`
- `pgen`   (length: ngen): active power of generators
- `qgen`   (length: ngen): reactive power of generators
- `pload`  (length: nbus): active loads
- `qload`  (length: nbus): reactive loads
- `dx`   (length: nstates): cache the difference between two consecutive states (used in power flow resolution)
- `balance` (length: nstates): cache for current power imbalance (used in power flow resolution)
- `bus_gen` (length: ngen): generator-bus incidence matrix `Cg`

"""
struct PolarNetworkState{VI,VT} <: AbstractNetworkBuffer
    vmag::VT
    vang::VT
    pnet::VT
    qnet::VT
    pgen::VT
    qgen::VT
    pload::VT
    qload::VT
    balance::VT
    dx::VT
    bus_gen::VI   # Generator-Bus incidence matrix
end

setvalues!(buf::PolarNetworkState, ::PS.VoltageMagnitude, values) = copyto!(buf.vmag, values)
setvalues!(buf::PolarNetworkState, ::PS.VoltageAngle, values) = copyto!(buf.vang, values)
function setvalues!(buf::PolarNetworkState, ::PS.ActivePower, values)
    pgenbus = view(buf.pnet, buf.bus_gen)
    pgenbus .= values
    copyto!(buf.pgen, values)
end
function setvalues!(buf::PolarNetworkState, ::PS.ReactivePower, values)
    qgenbus = view(buf.qnet, buf.bus_gen)
    qgenbus .= values
    copyto!(buf.qgen, values)
end
function setvalues!(buf::PolarNetworkState, ::PS.ActiveLoad, values)
    copyto!(buf.pload, values)
end
function setvalues!(buf::PolarNetworkState, ::PS.ReactiveLoad, values)
    copyto!(buf.qload, values)
end

function Base.iszero(buf::PolarNetworkState)
    return iszero(buf.pnet) &&
        iszero(buf.qnet) &&
        iszero(buf.vmag) &&
        iszero(buf.vang) &&
        iszero(buf.pgen) &&
        iszero(buf.qgen) &&
        iszero(buf.pload) &&
        iszero(buf.qload) &&
        iszero(buf.balance) &&
        iszero(buf.dx)
end

voltage(buf::PolarNetworkState) = buf.vmag .* exp.(im .* buf.vang)
voltage_host(buf) = voltage(buf) |> Array

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

function NetworkTopology(pf::PS.PowerNetwork, ::Type{VTI}, ::Type{VTD}) where {VTI, VTD}
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

