module PowerSystem

using ..ExaPF: Parse

using SparseArrays
using Printf

"""
    PowerNetwork

This structure contains constant parameters that define the topology and
physics of the power network.

The object is first created in main memory and then, if GPU computation is
enabled, some of the contents will be moved to the device.

"""
struct PowerNetwork
    V::Array{Complex{Float64}}
    Ybus::SparseArrays.SparseMatrixCSC{Complex{Float64},Int64}
    data::Dict{String,Array}

    nbus::Int64
    ngen::Int64
    nload::Int64

    bustype::Array{Int64}
    ref::Array{Int64}
    pv::Array{Int64}
    pq::Array{Int64}

    Sbus::Array{Complex{Float64}}

    function PowerNetwork(datafile::String)
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
        Ybus, Yf_br, Yt_br, Yf_tr, Yt_tr = makeYbus(data)

        # bus type indexing
        ref, pv, pq, bustype = bustypeindex(bus, gen)
    
        Sbus = assembleSbus(gen, load, SBASE, nbus)
        
        new(V, Ybus, data, nbus, ngen, nload, bustype, ref, pv, pq,
            Sbus)
    end

end

"""
    view(PowerNetwork)

Prints power network characteristics.

"""

function view(pf::PowerNetwork)
    println("Power Network characteristics:")
    @printf("\tBuses: %d. Slack: %d. PV: %d. PQ: %d\n", pf.nbus, length(pf.ref),
            length(pf.pv), length(pf.pq))
    println("\tGenerators: ", pf.ngen, ".")
    println("\tLoads: ", pf.nload, ".")

    # Print system status
    @printf("\t==============================================\n")
    @printf("\tBUS \t TYPE \t VMAG \t VANG \t P \t Q\n")
    @printf("\t==============================================\n")
    
    for i=1:pf.nbus
        type = pf.bustype[i]
        vmag = abs(pf.V[i])
        vang = angle(pf.V[i])*(360.0/pi)
        pinj = real(pf.Sbus[i])
        qinj = imag(pf.Sbus[i])
        @printf("\t%i \t  %d \t %1.3f\t%3.2f\t%3.3f\t%3.3f\n", i,
                type, vmag, vang, pinj, pinj)
    end

end




"""
    assemble_vecs(PowerNetwork)

Returns vectors x, u, p from network variables (VMAG, VANG, P and Q)
and bus type info.


Vector x is the variable vector, consisting on:
    - VMAG, VANG for buses of type PQ (type 1)
    - VANG for buses of type PV (type 2)
These variables are determined by the physics of the network.

Vector u is the control vector, consisting on:
    - VMAG, P for buses of type PV (type 1)
    - VM for buses of type SLACK (type 3)
These variables are controlled by the grid operator.

Vector p is the parameter vector, consisting on:
    - VA for buses of type SLACK (type 3)
    - P, Q for buses of type PQ (type 1)
These parameters are fixed through the problem.

In addition, variables:
    - Q for buses of type PV (type 2)
    - P, Q for buses of type SLACK (type 3)
are explicitly obtained from the previous variables, and hence
not needed in the computation.
"""
function assemble_vecs(pf::PowerNetwork)
 
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)
    println(nref, npv, npq)


    # build vector x
    dimension = 2*npq + npv
    x = zeros(dimension, 1)
    k = 1

    for bus=1:pf.nbus
        if pf.bustype[bus] == 1 #PQ
            x[k] = abs(pf.V[bus])
            x[k + 1] = angle(pf.V[bus])
            k = k + 2
        elseif pf.bustype[bus] == 2 #PV
            x[k] = angle(pf.V[bus])
            k = k + 1
        end
    end

    # build vector u
    dimension = 2*npv + nref
    u = zeros(dimension, 1)
    k = 1

    for bus=1:pf.nbus
        if pf.bustype[bus] == 2 #PV
            u[k] = abs(pf.V[bus])
            u[k + 1] = real(pf.Sbus[bus])
            k = k + 2
        elseif pf.bustype[bus] == 3 # slack
            u[k] = abs(pf.V[bus])
            k = k + 1
        end
    end

    # build vector p
    dimension = nref + 2*npq
    p = zeros(dimension, 1)
    k = 1
    
    for bus=1:pf.nbus
        if pf.bustype[bus] == 3 #slack
            p[k] = abs(pf.V[bus])
            k = k + 1
        elseif pf.bustype[bus] == 1 # PQ
            p[k] = real(pf.Sbus[bus])
            p[k + 1] = imag(pf.Sbus[bus])
            k = k + 2
        end
    end

    return x, u, p

end


"""
    xup_to_vs(PowerNetwork, x, u, p)

Converts x, u, p vectors to vmag, vang, pinj and qinj.
"""
function xup_to_vs(pf::PowerNetwork, x::Array{Float64}, u::Array{Float64},
                   p::Array{Float64})

    nbus = pf.nbus
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

    vmag = zeros(nbus)
    vang = zeros(nbus)
    pinj = zeros(nbus)
    qinj = zeros(nbus)

    x_idx = 1
    u_idx = 1
    p_idx = 1

    for bus=1:nbus
        if pf.bustype[bus] == 1 #PQ
            vmag[bus] = x[x_idx]
            vang[bus] = x[x_idx + 1]
            pinj[bus] = p[p_idx]
            qinj[bus] = p[p_idx + 1]
            x_idx += 2
            p_idx += 2
        elseif pf.bustype[bus] == 2 #PV
            vmag[bus] = u[u_idx]
            pinj[bus] = u[u_idx + 1]
            vang[bus] = x[x_idx]
            u_idx += 2
            x_idx += 1
        elseif pf.bustype[bus] == 3 # ref
            vmag[bus] = u[u_idx]
            vang[bus] = p[p_idx]
            u_idx += 1
            p_idx += 1
        end

    end


    return vmag, vang, pinj, qinj

end






"""
    bustypeindex(bus, gen)

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

    return ref, pv, pq, bustype
end


"""
    assembleSbus(gen, load, SBASE, nbus)

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

# Create an admittance matrix. The implementation is a modification of
# MATPOWER's makeYbus. We attach the original MATPOWER's license in makeYbus.m:
#
# MATPOWER
# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC)
# by Ray Zimmerman, PSERC Cornell
#
# Covered by the 3-clause BSD License.
#
# This function returns the following:
#
#  Ybus : nb  x nb admittance
#  Yf_br: nbr x nb from-bus admittance of non-transformer branches
#  Yt_br: nbr x nb to-bus admittance of non-transformer branches
#  Yf_tr: ntr x nb from-bus admittance of transformer branches
#  Yt_tr: ntr x nb to-bus admittance of transformer branches
#
# where nb is the number of buses, nbr is the number of non-transformer
# branches, and ntr is the number of transformer branches.

function makeYbus(raw_data)
    baseMVA = raw_data["CASE IDENTIFICATION"][1]
    bus = raw_data["BUS"]
    branch = raw_data["BRANCH"]
    trans = raw_data["TRANSFORMER"]
    fsh = raw_data["FIXED SHUNT"]

    BUS_B, BUS_AREA, BUS_VM, BUS_VA, BUS_NVHI, BUS_NVLO, BUS_EVHI,
        BUS_EVLO = Parse.idx_bus()
    BR_FR, BR_TO, BR_CKT, BR_R, BR_X, BR_B, BR_RATEA, BR_RATEC,
        BR_STAT = Parse.idx_branch()
    TR_FR, TR_TO, TR_CKT, TR_MAG1, TR_MAG2, TR_STAT, TR_R, TR_X, TR_WINDV1,
        TR_ANG, TR_RATEA, TR_RATEC, TR_WINDV2 = Parse.idx_transformer()
    FSH_BUS, FSH_ID, FSH_STAT, FSH_G, FSH_B = Parse.idx_fshunt()

    nb = size(bus, 1)
    nbr = size(branch, 1)
    ntr = size(trans, 1)

    i2b = bus[:, BUS_B]
    b2i = sparse(i2b, ones(nb), collect(1:nb), maximum(i2b), 1)

    st_br = branch[:, BR_STAT]
    Ys_br = st_br ./ (branch[:, BR_R] .+ im*branch[:, BR_X])
    B_br = st_br .* branch[:, BR_B]
    Ytt_br = Ys_br + im*B_br/2
    Yff_br = Ytt_br
    Yft_br = -Ys_br
    Ytf_br = -Ys_br

    f = [b2i[b] for b in branch[:, BR_FR]]
    t = [b2i[b] for b in branch[:, BR_TO]]
    i = collect(1:nbr)
    Cf_br = sparse(i, f, ones(nbr), nbr, nb)
    Ct_br = sparse(i, t, ones(nbr), nbr, nb)
    Yf_br = sparse(i, i, Yff_br, nbr, nbr) * Cf_br +
            sparse(i, i, Yft_br, nbr, nbr) * Ct_br
    Yt_br = sparse(i, i, Ytf_br, nbr, nbr) * Cf_br +
            sparse(i, i, Ytt_br, nbr, nbr) * Ct_br

    st_tr = trans[:, TR_STAT]
    Ys_tr = st_tr ./ (trans[:, TR_R] .+ im*trans[:, TR_X])
    tap = (trans[:, TR_WINDV1] ./ trans[:, TR_WINDV2]) .* exp.(im*pi/180 .* trans[:, TR_ANG])
    Ymag = st_tr .* (trans[:, TR_MAG1] .+ im*trans[:, TR_MAG2])
    Ytt_tr = Ys_tr
    Yff_tr = (Ytt_tr ./ (tap .* conj(tap))) .+ Ymag
    Yft_tr = -Ys_tr ./ conj(tap)
    Ytf_tr = -Ys_tr ./ tap

    f = [b2i[b] for b in trans[:, TR_FR]]
    t = [b2i[b] for b in trans[:, TR_TO]]
    i = collect(1:ntr)
    Cf_tr = sparse(i, f, ones(ntr), ntr, nb)
    Ct_tr = sparse(i, t, ones(ntr), ntr, nb)
    Yf_tr = sparse(i, i, Yff_tr, ntr, ntr) * Cf_tr +
            sparse(i, i, Yft_tr, ntr, ntr) * Ct_tr
    Yt_tr = sparse(i, i, Ytf_tr, ntr, ntr) * Cf_tr +
            sparse(i, i, Ytt_tr, ntr, ntr) * Ct_tr

    Ysh = zeros(Complex{Float64}, nb)
    for i=1:size(fsh, 1)
        Ysh[b2i[fsh[i, FSH_BUS]]] += fsh[i, FSH_STAT] * (fsh[i, FSH_G] + im*fsh[i, FSH_B])/baseMVA
    end

    Ybus = Cf_br' * Yf_br + Ct_br' * Yt_br +  # branch admittances
           Cf_tr' * Yf_tr + Ct_tr' * Yt_tr +  # transformer admittances
           sparse(1:nb, 1:nb, Ysh, nb, nb)    # shunt admittances

    return Ybus, Yf_br, Yt_br, Yf_tr, Yt_tr
end

end
