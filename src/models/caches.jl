abstract type AbstractCache end

"Store indexing on target device"
struct IndexingCache{IVT} <: AbstractCache
    index_pv::IVT
    index_pq::IVT
    index_ref::IVT
    index_generators::IVT
end

