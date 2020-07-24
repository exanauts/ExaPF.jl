module PowerSystem

using ..ExaPF: Parse
using ..ExaPF: Network

using SparseArrays

"""
    Pf

This structure contains constant parameters that define the topology and
physics of the power network.

# Fields
- `V::Array{Complex{Float64}}`: voltage
"""
struct Pf
    V::Array{Complex{Float64}}
    Ybus::SparseArrays.SparseMatrixCSC{Complex{Float64},Int64}
    data::Dict{String,Array}
    
    nbus::Int64
    ngen::Int64
    nload::Int64

    ref::Array{Int64}
    pv::Array{Int64}
    pq::Array{Int64}
    
    Sbus::Array{Complex{Float64}}

    function Pf(datafile::String)
        data = Parse.parse_raw(datafile)
        
        # Parsed data indexes
        BUS_B, BUS_AREA, BUS_VM, BUS_VA, BUS_NVHI, BUS_NVLO, BUS_EVHI,
        BUS_EVLO, BUS_TYPE = Parse.idx_bus()
        GEN_BUS, GEN_ID, GEN_PG, GEN_QG, GEN_QT, GEN_QB, GEN_STAT,
        GEN_PT, GEN_PB = Parse.idx_gen()
        LOAD_BUS, LOAD_ID, LOAD_STATUS, LOAD_PL, LOAD_QL = Parse.idx_load()

        # retrive required data
        bus = data["BUS"]
        gen = data["GENERATOR"]
        load = data["LOAD"]
        SBASE = data["CASE IDENTIFICATION"][1]

        # size of the system
        nbus = size(bus, 1)
        ngen = size(gen, 1)
        nload = size(load, 1)

        # obtain V0 from raw data
        V = Array{Complex{Float64}}(undef, nbus)
        for i in 1:nbus
            V[i] = bus[i, BUS_VM]*exp(1im * pi/180 * bus[i, BUS_VA])
        end
        
        # form Y matrix
        Ybus, Yf_br, Yt_br, Yf_tr, Yt_tr = Network.makeYbus(data)
        
        # bus type indexing
        ref, pv, pq = bustypeindex(bus, gen)
    
        Sbus = assembleSbus(gen, load, SBASE, nbus)
        
        new(V, Ybus, data, nbus, ngen, nload, ref, pv, pq, Sbus)
    end
end


"""
bustypeindex(data)

Returns vectors indexing buses by type: ref, pv, pq.

"""
function bustypeindex(bus, gen)
    # retrieve indeces
    BUS_B, BUS_AREA, BUS_VM, BUS_VA, BUS_NVHI, BUS_NVLO, BUS_EVHI,
    BUS_EVLO, BUS_TYPE = Parse.idx_bus()

    GEN_BUS, GEN_ID, GEN_PG, GEN_QG, GEN_QT, GEN_QB, GEN_STAT,
    GEN_PT, GEN_PB = Parse.idx_gen()

    # form vector that lists the number of generators per bus.
    # If a PV bus has 0 generators (e.g. due to contingency)
    # then that bus turns to a PQ bus.

    # Design note: this might be computed once and then modified for each contingency.
    gencon = zeros(Int8, size(bus, 1))

    for i in 1:size(gen, 1)
        if gen[i, GEN_STAT] == 1
            gencon[gen[i, GEN_BUS]] += 1
        end
    end

    bustype = copy(bus[:, BUS_TYPE])

    for i in 1:size(bus, 1)
        if (bustype[i] == 2) && (gencon[i] == 0)
            bustype[i] = 1
        elseif (bustype[i] == 1) && (gencon[i] > 0)
            bustype[i] = 2
        end
    end

    # form vectors
    ref = findall(x->x==3, bustype)
    pv = findall(x->x==2, bustype)
    pq = findall(x->x==1, bustype)

    return ref, pv, pq
end


"""
assembleSbus(data)

Assembles vector of constant power injections (generator - load). Since
we do not have voltage-dependent loads, this vector only needs to be
assembled once at the beginning of the power flow routine.

"""
function assembleSbus(gen, load, SBASE, nbus)
    Sbus = zeros(Complex{Float64}, nbus)

    ngen = size(gen, 1)
    nload = size(load, 1)

    # retrieve indeces
    GEN_BUS, GEN_ID, GEN_PG, GEN_QG, GEN_QT, GEN_QB, GEN_STAT,
    GEN_PT, GEN_PB = Parse.idx_gen()

    LOAD_BUS, LOAD_ID, LOAD_STAT, LOAD_PL, LOAD_QL = Parse.idx_load()

    for i in 1:ngen
        if gen[i, GEN_STAT] == 1
            Sbus[gen[i, GEN_BUS]] += (gen[i, GEN_PG] + 1im*gen[i, GEN_QG])/SBASE
        end
    end

    for i in 1:nload
        if load[i, LOAD_STAT] == 1
            Sbus[load[i, LOAD_BUS]] -= (load[i, LOAD_PL] + 1im*load[i, LOAD_QL])/SBASE
        end
    end

    return Sbus
end

end
