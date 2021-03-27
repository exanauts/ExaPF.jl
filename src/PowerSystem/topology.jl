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
    ANGMIN, ANGMAX, PF, QF, PT, QT, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX = IndexSet.idx_branch()
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()

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

    # build Ybus
    Ybus = Cf' * Yf + Ct' * Yt + sparse(1:nb, 1:nb, Ysh, nb, nb)

    return Ybus

end

"""
    bustypeindex(bus, gen)

Returns vectors indexing buses by type: ref, pv, pq.

"""
function bustypeindex(bus, gen, bus_to_indexes)
    # retrieve indeces
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()

    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAX, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()


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
        if (bustype[i] == PV_BUS_TYPE) && (gencon[i] == 0)
            bustype[i] = PQ_BUS_TYPE
        elseif (bustype[i] == PQ_BUS_TYPE) && (gencon[i] > 0)
            bustype[i] = PV_BUS_TYPE
        end
    end

    # form vectors
    ref = findall(x -> x==REF_BUS_TYPE, bustype)
    pv = findall(x -> x==PV_BUS_TYPE, bustype)
    pq = findall(x -> x==PQ_BUS_TYPE, bustype)

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
    sbus = zeros(Complex{Float64}, nbus)
    sload = zeros(Complex{Float64}, nbus)

    # retrieve indeces
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()

    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAX, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()

    for i in 1:ngen
        if gen[i, GEN_STATUS] == 1
            id_bus = bus_to_indexes[gen[i, GEN_BUS]]
            sbus[id_bus] += (gen[i, PG] + 1im*gen[i, QG])/baseMVA
        end
    end

    for i in 1:nbus
        id_bus = bus_to_indexes[bus[i, BUS_I]]
        load = (bus[i, PD] + 1im*bus[i, QD])/baseMVA
        sbus[id_bus] -= load
        sload[id_bus] = load
    end

    return sbus, sload
end
