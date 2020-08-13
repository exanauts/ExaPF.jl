module PowerSystem

using ..ExaPF: ParsePSSE, ParseMAT, IdxSet

import Base: show
using Printf
using SparseArrays

const PQ_BUS_TYPE = 1
const PV_BUS_TYPE = 2
const REF_BUS_TYPE  = 3

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

    bustype::Array{Int64}
    bus_to_indexes::Dict{Int, Int}
    ref::Array{Int64}
    pv::Array{Int64}
    pq::Array{Int64}

    Sbus::Array{Complex{Float64}}

    function PowerNetwork(datafile::String, data_format::Int64=0)

        if data_format == 0
            println("Reading PSSE format")
            data_raw = ParsePSSE.parse_raw(datafile)
            data, bus_id_to_indexes = ParsePSSE.raw_to_exapf(data_raw)
        elseif data_format == 1
            data_mat = ParseMAT.parse_mat(datafile)
            data, bus_id_to_indexes = ParseMAT.mat_to_exapf(data_mat)
        end
        # Parsed data indexes
        BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
        LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IdxSet.idx_bus()

        # retrive required data
        bus = data["bus"]
        gen = data["gen"]
        SBASE = data["baseMVA"][1]

        # size of the system
        nbus = size(bus, 1)
        ngen = size(gen, 1)

        # obtain V0 from raw data
        V = Array{Complex{Float64}}(undef, nbus)
        for i in 1:nbus
            V[i] = bus[i, VM]*exp(1im * pi/180 * bus[i, VA])
        end

        # form Y matrix
        Ybus = makeYbus(data, bus_id_to_indexes)

        # bus type indexing
        ref, pv, pq, bustype = bustypeindex(bus, gen, bus_id_to_indexes)

        Sbus = assembleSbus(gen, bus, SBASE, bus_id_to_indexes)

        new(V, Ybus, data, nbus, ngen, bustype, bus_id_to_indexes, ref, pv, pq, Sbus)
    end

end

function Base.show(io::IO, pf::PowerNetwork)
    println("Power Network characteristics:")
    @printf("\tBuses: %d. Slack: %d. PV: %d. PQ: %d\n", pf.nbus, length(pf.ref),
            length(pf.pv), length(pf.pq))
    println("\tGenerators: ", pf.ngen, ".")
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
                type, vmag, vang, pinj, qinj)
    end

end

"""
    get_x(PowerNetwork)

Returns vector x from network variables (VMAG, VANG, P and Q)
and bus type info.

Vector x is the variable vector, consisting on:
    - VMAG, VANG for buses of type PQ (type 1)
    - VANG for buses of type PV (type 2)
These variables are determined by the physics of the network.

Ordering:

x = [VMAG^{PQ}, VANG^{PQ}, VANG^{PV}]
"""
function get_x(pf::PowerNetwork)

    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

    # build vector x
    dimension = 2*npq + npv
    x = zeros(dimension)

    x[1:npq] = abs.(pf.V[pf.pq])
    x[npq + 1:2*npq] = angle.(pf.V[pf.pq])
    x[2*npq + 1:2*npq + npv] = angle.(pf.V[pf.pv])

    return x
end

"""
    get_u(PowerNetwork)

Returns vector x from network variables (VMAG, VANG, P and Q)
and bus type info.

Vector u is the control vector, consisting on:
    - VMAG, P for buses of type PV (type 1)
    - VM for buses of type SLACK (type 3)
These variables are controlled by the grid operator.

Ordering:

u = [VMAG^{REF}, P^{PV}, V^{PV}]
"""
function get_u(pf::PowerNetwork)

    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

    # build vector u
    dimension = 2*npv + nref
    u = zeros(dimension)

    u[1:nref] = abs.(pf.V[pf.ref])
    u[nref + 1:nref + npv] = real.(pf.Sbus[pf.pv])
    u[nref + npv + 1:nref + 2*npv] = abs.(pf.V[pf.pv])

    return u
end

"""
    get_p(PowerNetwork)

Returns vector p from network variables (VMAG, VANG, P and Q)
and bus type info.

Vector p is the parameter vector, consisting on:
    - VA for buses of type SLACK (type 3)
    - P, Q for buses of type PQ (type 1)
These parameters are fixed through the problem.

Order:

p = [vang^{ref}, p^{pq}, q^{pq}]
"""
function get_p(pf::PowerNetwork)

    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

    # build vector p
    dimension = nref + 2*npq
    p = zeros(dimension)

    p[1:nref] = angle.(pf.V[pf.ref])
    p[nref + 1:nref + npq] = real.(pf.Sbus[pf.pq])
    p[nref + npq + 1:nref + 2*npq] = imag.(pf.Sbus[pf.pq])

    return p
end

"""
    retrieve_physics(PowerNetwork, x, u, p)

Converts x, u, p vectors to vmag, vang, pinj and qinj.
"""
function retrieve_physics(
    pf::PowerNetwork,
    x::VT,
    u::VT,
    p::VT,
) where {T<:Real, VT<:AbstractVector{T}}

    nbus = pf.nbus
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

    vmag = zeros(nbus)
    vang = zeros(nbus)
    pinj = zeros(nbus)
    qinj = zeros(nbus)

    vmag[pf.pq] = x[1:npq]
    vang[pf.pq] = x[npq + 1:2*npq]
    vang[pf.pv] = x[2*npq + 1:2*npq + npv]

    vmag[pf.ref] = u[1:nref]
    pinj[pf.pv] = u[nref + 1:nref + npv]
    vmag[pf.pv] = u[nref + npv + 1:nref + 2*npv]

    vang[pf.ref] = p[1:nref]
    pinj[pf.pq] = p[nref + 1:nref + npq]
    qinj[pf.pq] = p[nref + npq + 1:nref + 2*npq]

    return vmag, vang, pinj, qinj
end

"""
    bustypeindex(bus, gen)

Returns vectors indexing buses by type: ref, pv, pq.

"""
function bustypeindex(bus, gen, bus_to_indexes)
    # retrieve indeces
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IdxSet.idx_bus()

    GEN_BUS, PG, QG, QMAX, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IdxSet.idx_gen()


    # form vector that lists the number of generators per bus.
    # If a PV bus has 0 generators (e.g. due to contingency)
    # then that bus turns to a PQ bus.

    # Design note: this might be computed once and then modified for each contingency.
    gencon = zeros(Int8, size(bus, 1))

    for i in 1:size(gen, 1)
        if gen[i, GEN_STATUS] == 1
            id_bus = bus_to_indexes[gen[i, GEN_BUS]]
            gencon[id_bus] += 1
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
function assembleSbus(gen, bus, baseMVA, bus_to_indexes)

    ngen = size(gen, 1)
    nbus = size(bus, 1)
    Sbus = zeros(Complex{Float64}, nbus)

    # retrieve indeces
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IdxSet.idx_bus()

    GEN_BUS, PG, QG, QMAX, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IdxSet.idx_gen()

    for i in 1:ngen
        if gen[i, GEN_STATUS] == 1
            id_bus = bus_to_indexes[gen[i, GEN_BUS]]
            Sbus[id_bus] += (gen[i, PG] + 1im*gen[i, QG])/baseMVA
        end
    end

    for i in 1:nbus
        id_bus = bus_to_indexes[bus[i, BUS_I]]
        Sbus[id_bus] -= (bus[i, PD] + 1im*bus[i, QD])/baseMVA
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
function makeYbus(data, bus_to_indexes)
    baseMVA = data["baseMVA"]
    bus     = data["bus"]
    branch  = data["branch"]
    F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS,
    ANGMIN, ANGMAX, PF, QF, PT, QT, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX = IdxSet.idx_branch()
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IdxSet.idx_bus()

    # constants
    nb = size(bus, 1)          # number of buses
    nl = size(branch, 1)       # number of lines

    # define named indices into bus, branch matrices

    # for each branch, compute the elements of the branch admittance matrix where
    #
    #      | If |   | Yff  Yft |   | Vf |
    #      |    | = |          | * |    |
    #      | It |   | Ytf  Ytt |   | Vt |
    #
    stat = branch[:, BR_STATUS]                     # ones at in-service branches
    Ys = stat ./ (branch[:, BR_R] + 1im * branch[:, BR_X])  # series admittance
    Bc = stat .* branch[:, BR_B]                           # line charging susceptance
    tap = ones(nl)                               # default tap ratio = 1
    i = findall(branch[:, TAP] .!= 0)              # indices of non-zero tap ratios
    tap[i] = branch[i, TAP]                         # assign non-zero tap ratios
    tap = tap .* exp.(1im*pi/180 * branch[:, SHIFT]) # add phase shifters
    Ytt = Ys + 1im*Bc/2
    Yff = Ytt ./ (tap .* conj(tap))
    Yft = - Ys ./ conj(tap)
    Ytf = - Ys ./ tap

    # compute shunt admittance
    # if Psh is the real power consumed by the shunt at V = 1.0 p.u.
    # and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
    # then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    # i.e. Ysh = Psh + j Qsh, so ...
    Ysh = (bus[:, GS] + 1im * bus[:, BS]) / baseMVA[1] # vector of shunt admittances

    # build connection matrices
    f = [bus_to_indexes[e] for e in branch[:, F_BUS]] # list of "from" buses
    t = [bus_to_indexes[e] for e in branch[:, T_BUS]] # list of "to" buses

    Cf = sparse(1:nl, f, ones(nl), nl, nb)       # connection matrix for line & from buses
    Ct = sparse(1:nl, t, ones(nl), nl, nb)       # connection matrix for line & to buses

    # build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    # at each branch's "from" bus, and Yt is the same for the "to" bus end
    i = [1:nl; 1:nl]
    Yf = sparse(i, [f; t], [Yff; Yft], nl, nb)
    Yt = sparse(i, [f; t], [Ytf; Ytt], nl, nb)

    # Yf = spdiags(Yff, 0, nl, nl) * Cf + spdiags(Yft, 0, nl, nl) * Ct;
    # Yt = spdiags(Ytf, 0, nl, nl) * Cf + spdiags(Ytt, 0, nl, nl) * Ct;

    # build Ybus
    Ybus = Cf' * Yf + Ct' * Yt + sparse(1:nb, 1:nb, Ysh, nb, nb)

    return Ybus

end

end
