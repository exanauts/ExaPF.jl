# Power flow module. The implementation is a modification of
# MATPOWER's code. We attach the original MATPOWER's license in makeYbus.m:
#
# MATPOWER
# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC)
# by Ray Zimmerman, PSERC Cornell
#
# Covered by the 3-clause BSD License.
#
# Expression of Jacobians and Hessians from MATPOWER

#
function matpower_residual_jacobian(V, Ybus)
    n = size(V, 1)
    Ibus = Ybus*V
    diagV       = sparse(1:n, 1:n, V, n, n)
    diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
    diagVnorm   = sparse(1:n, 1:n, V./abs.(V), n, n)

    dSbus_dVm = diagV * conj(Ybus * diagVnorm) .+ conj(diagIbus) * diagVnorm
    dSbus_dVa = 1im * diagV * conj(diagIbus .- Ybus * diagV)
    return (dSbus_dVm, dSbus_dVa)
end

function _matpower_lineflow_jacobian(V, branches::Branches)
    nb = size(V, 1)
    f = branches.from_buses
    t = branches.to_buses
    nl = length(f)

    diagV     = sparse(1:nb, 1:nb, V, nb, nb)
    diagVnorm = sparse(1:nb, 1:nb, V./abs.(V), nb, nb)

    # Connection matrices
    Cf = sparse(1:nl, f, ones(nl), nl, nb)
    Ct = sparse(1:nl, t, ones(nl), nl, nb)

    i = [1:nl; 1:nl]
    Yf = sparse(i, [f; t], [branches.Yff; branches.Yft], nl, nb)
    Yt = sparse(i, [f; t], [branches.Ytf; branches.Ytt], nl, nb)

    If = Yf * V
    It = Yt * V

    diagCfV = sparse(1:nl, 1:nl, Cf * V, nl, nl)
    diagCtV = sparse(1:nl, 1:nl, Ct * V, nl, nl)

    Sf = diagCfV * conj(If)
    St = diagCtV * conj(It)

    diagIf = sparse(1:nl, 1:nl, If, nl, nl)
    diagIt = sparse(1:nl, 1:nl, It, nl, nl)

    dSf_dVm = conj(diagIf) * Cf * diagVnorm + diagCfV * conj(Yf * diagVnorm)
    dSf_dVa = 1im * (conj(diagIf) * Cf * diagV - diagCfV * conj(Yf * diagV))

    dSt_dVm = conj(diagIt) * Ct * diagVnorm + diagCtV * conj(Yt * diagVnorm)
    dSt_dVa = 1im * (conj(diagIt) * Ct * diagV - diagCtV * conj(Yt * diagV))

    return (Sf, St, dSf_dVm, dSf_dVa, dSt_dVm, dSt_dVa)
end

function matpower_lineflow_power_jacobian(V, branches::Branches)
    nl = length(branches.from_buses)
    (Sf, St, dSf_dVm, dSf_dVa, dSt_dVm, dSt_dVa) = _matpower_lineflow_jacobian(V, branches)

    dSf = sparse(1:nl, 1:nl, Sf, nl, nl)
    dSt = sparse(1:nl, 1:nl, St, nl, nl)

    dHf_dVm = 2 * (real(dSf) * real(dSf_dVm) + imag(dSf) * imag(dSf_dVm))
    dHf_dVa = 2 * (real(dSf) * real(dSf_dVa) + imag(dSf) * imag(dSf_dVa))

    dHt_dVm = 2 * (real(dSt) * real(dSt_dVm) + imag(dSt) * imag(dSt_dVm))
    dHt_dVa = 2 * (real(dSt) * real(dSt_dVa) + imag(dSt) * imag(dSt_dVa))

    dH_dVm = [dHf_dVm; dHt_dVm]
    dH_dVa = [dHf_dVa; dHt_dVa]
    return dH_dVm, dH_dVa
end

# Hessian vector-product H*λ, H = Jₓₓ (from Matpower)
# Suppose ordering is correct
function _matpower_hessian(V, Ybus, λ)
    n = size(V, 1)
    # build up auxiliary matrices
    Ibus = Ybus * V
    diagV       = sparse(1:n, 1:n, V, n, n)
    diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
    d_invV      = sparse(1:n, 1:n, 1 ./ abs.(V), n, n)
    d_lambda    = sparse(1:n, 1:n, λ, n, n)

    A = sparse(1:n, 1:n, λ .* V, n, n)
    B = Ybus * diagV
    C = A * conj(B)
    D = Ybus' * diagV
    Dl = sparse(1:n, 1:n, D * λ, n, n)
    E = conj(diagV)  * (D * d_lambda - Dl)
    F = C - A * sparse(1:n, 1:n, conj(Ibus), n, n)
    G = d_invV

    # θθ
    G11 = E + F
    # vθ
    G21 = 1im .* G * (E - F)
    # vv
    G22 = G * (C + transpose(C)) * G
    return (G11, transpose(G21), G22)
end

function residual_hessian(V, Ybus, λ, pv, pq, ref)
    # decompose vector
    n = length(V)
    λp = zeros(n) ; λq = zeros(n)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)
    λp[pv] = λ[1:npv]
    λp[pq] = λ[npv+1:npv+npq]
    λq[pq] = λ[npv+npq+1:end]

    Gp11, Gp12, Gp22 = _matpower_hessian(V, Ybus, λp)
    Pθθ = real.(Gp11)
    Pvθ = real.(Gp12)
    Pvv = real.(Gp22)

    Gq11, Gq12, Gq22 = _matpower_hessian(V, Ybus, λq)
    Qθθ = imag.(Gq11)
    Qvθ = imag.(Gq12)
    Qvv = imag.(Gq22)

    # w.r.t. xx
    H11 = Pθθ[pv, pv] + Qθθ[pv, pv]
    H12 = Pθθ[pv, pq] + Qθθ[pv, pq]
    H13 = Pvθ[pv, pq] + Qvθ[pv, pq]
    H22 = Pθθ[pq, pq] + Qθθ[pq, pq]
    H23 = Pvθ[pq, pq] + Qvθ[pq, pq]
    H33 = Pvv[pq, pq] + Qvv[pq, pq]

    Hxx = [
        H11  H12  H13
        H12' H22  H23
        H13' H23' H33
    ]::SparseMatrixCSC{Float64, Int}

    # w.r.t. uu
    H11 = Pvv[ref, ref] + Qvv[ref, ref]
    H12 = Pvv[ref,  pv] + Qvv[ref,  pv]
    H22 = Pvv[pv,   pv] + Qvv[pv,   pv]

    Huu = [
         H11  H12 spzeros(nref, npv)
         H12' H22 spzeros(npv, npv)
         spzeros(npv, nref + 2 * npv)
    ]::SparseMatrixCSC{Float64, Int}

    # w.r.t. xu
    Pvθ = real.(transpose(Gp12))
    Qvθ = imag.(transpose(Gq12))
    H11 = Pvθ[ref, pv] + Qvθ[ref, pv]
    H12 = Pvθ[ref, pq] + Qvθ[ref, pq]
    H13 = Pvv[ref, pq] + Qvv[ref, pq]
    H21 = Pvθ[pv,  pv] + Qvθ[pv,  pv]
    H22 = Pvθ[pv,  pq] + Qvθ[pv,  pq]
    H23 = Pvv[pv,  pq] + Qvv[pv,  pq]

    Hxu = [
        H11  H12  H13
        H21  H22  H23
        spzeros(npv, npv + 2 * npq)
    ]::SparseMatrixCSC{Float64, Int}

    return (Hxx, Hxu, Huu)
end

# ∂²pg_ref / ∂²x
function active_power_hessian(V, Ybus, pv, pq, ref)
    # decompose vector
    n = length(V)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    λp = zeros(n)
    # Pick only components wrt ref nodes
    λp[ref] .= 1.0

    G11, G12, G22 = _matpower_hessian(V, Ybus, λp)
    Pθθ = real.(G11)
    Pvθ = real.(G12)
    Pvv = real.(G22)

    H11 = Pθθ[pv, pv]
    H12 = Pθθ[pv, pq]
    H13 = Pvθ[pv, pq]
    H22 = Pθθ[pq, pq]
    H23 = Pvθ[pq, pq]
    H33 = Pvv[pq, pq]

    # w.r.t. xx
    Hxx = [
        H11  H12  H13
        H12' H22  H23
        H13' H23' H33
    ]::SparseMatrixCSC{Float64, Int}

    # w.r.t. uu
    H11 = Pvv[ref, ref]
    H12 = Pvv[ref, pv]
    H22 = Pvv[pv, pv]

    Huu = [
         H11  H12 spzeros(nref, npv)
         H12' H22 spzeros(npv, npv)
         spzeros(npv, nref + 2 * npv)
    ]::SparseMatrixCSC{Float64, Int}

    # w.r.t. xu
    Pvθ = real.(transpose(G12))
    Qvθ = imag.(transpose(G12))
    H11 = Pvθ[ref, pv]
    H12 = Pvθ[ref, pq]
    H13 = Pvv[ref, pq]
    H21 = Pvθ[pv, pv]
    H22 = Pvθ[pv, pq]
    H23 = Pvv[pv, pq]

    Hxu = [
        H11  H12  H13
        H21  H22  H23
        spzeros(npv, npv + 2 * npq)
    ]::SparseMatrixCSC{Float64, Int}

    return (Hxx, Hxu, Huu)
end

# ∂²qg / ∂²x * λ
function reactive_power_hessian(V, Ybus, λ, pv, pq, ref)
    n = length(V)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    G11, G12, G22 = _matpower_hessian(V, Ybus, λ)
    Qθθ = imag.(G11)
    Qvθ = imag.(G12)
    Qvv = imag.(G22)

    H11 = Qθθ[pv, pv]
    H12 = Qθθ[pv, pq]
    H13 = Qvθ[pv, pq]
    H22 = Qθθ[pq, pq]
    H23 = Qvθ[pq, pq]
    H33 = Qvv[pq, pq]

    Hxx = [
        H11  H12  H13
        H12' H22  H23
        H13' H23' H33
    ]::SparseMatrixCSC{Float64, Int}

    H11 = Qvv[ref, ref]
    H12 = Qvv[ref, pv]
    H22 = Qvv[pv, pv]

    Huu = [
         H11  H12 spzeros(nref, npv)
         H12' H22 spzeros(npv, npv)
         spzeros(npv, nref + 2* npv)
    ]::SparseMatrixCSC{Float64, Int}

    Pvθ = real.(transpose(G12))
    Qvθ = imag.(transpose(G12))
    H11 = Qvθ[ref, pv]
    H12 = Qvθ[ref, pq]
    H13 = Qvv[ref, pq]
    H21 = Qvθ[pv, pv]
    H22 = Qvθ[pv, pq]
    H23 = Qvv[pv, pq]

    Hxu = [
        H11  H12  H13
        H21  H22  H23
        spzeros(npv, 2*npq+npv)
    ]::SparseMatrixCSC{Float64, Int}
    return (Hxx, Hxu, Huu)
end

