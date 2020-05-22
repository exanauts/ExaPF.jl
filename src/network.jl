module Network

using SparseArrays
include("parse/parse.jl")
include("parse/parse_raw.jl")

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
        BUS_EVLO = idx_bus()
    BR_FR, BR_TO, BR_CKT, BR_R, BR_X, BR_B, BR_RATEA, BR_RATEC,
        BR_STAT = idx_branch()
    TR_FR, TR_TO, TR_CKT, TR_MAG1, TR_MAG2, TR_STAT, TR_R, TR_X, TR_WINDV1,
        TR_ANG, TR_RATEA, TR_RATEC, TR_WINDV2 = idx_transformer()
    FSH_BUS, FSH_ID, FSH_STAT, FSH_G, FSH_B = idx_fshunt()

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
