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

struct PolarNetworkState{VT} <: AbstractNetworkBuffer
    vmag::VT
    vang::VT
    pinj::VT
    qinj::VT
    pg::VT
    qg::VT
    balance::VT
    dx::VT
end

setvalues!(buf::PolarNetworkState, ::PS.VoltageMagnitude, values) = copyto!(buf.vmag, values)
setvalues!(buf::PolarNetworkState, ::PS.VoltageAngle, values) = copyto!(buf.vang, values)
setvalues!(buf::PolarNetworkState, ::PS.ActivePower, values) = copyto!(buf.pg, values)
setvalues!(buf::PolarNetworkState, ::PS.ReactivePower, values) = copyto!(buf.qg, values)
