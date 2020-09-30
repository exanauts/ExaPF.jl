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

function PolarNetworkState(nbus, ngen, device)
    if isa(device, CPU)
        VT = Vector{Float64}
    elseif isa(device, CUDADevice)
        VT = CuVector{Float64, Nothing}
    end
    vmag = VT(undef, nbus)
    vang = VT(undef, nbus)
    pinj = VT(undef, nbus)
    qinj = VT(undef, nbus)

    pg = VT(undef, ngen)
    qg = VT(undef, ngen)
    balance = VT(undef, 2*nbus)
    dx = VT(undef, 2*nbus)
    return PolarNetworkState{VT}(vmag, vang, pinj, qinj, pg, qg, balance, dx)
end

