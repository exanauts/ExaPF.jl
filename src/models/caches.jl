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

function PolarNetworkState(state::PolarNetworkState, device = nothing)
    if device == CUDADevice()
        VT = CuVector{Float64}
    elseif device == CPU()
        VT = Vector{Float64}
    else
        VT = typeof(state.vmag)
    end
    return PolarNetworkState{VT}(
        state.vmag,
        state.vang,
        state.pinj,
        state.qinj,
        state.pg,
        state.qg,
        state.balance,
        state.dx
        )
end

