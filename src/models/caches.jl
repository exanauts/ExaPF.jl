abstract type AbstractCache end

"Store indexing on target device"
struct IndexingCache{IVT} <: AbstractCache
    index_pv::IVT
    index_pq::IVT
    index_ref::IVT
    index_generators::IVT
end

struct NetworkState{VT} <: AbstractCache
    vmag::VT
    vang::VT
    pinj::VT
    qinj::VT
    pg::VT
    qg::VT
end

function NetworkState(nbus, ngen, device)
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
    return NetworkState{VT}(vmag, vang, pinj, qinj, pg, qg)
end

