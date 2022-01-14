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

    dSbus_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
    dSbus_dVa = 1im * diagV * conj(diagIbus - Ybus * diagV)
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

function _matpower_basis_jacobian(V, branches::Branches)
    nb = size(V, 1)
    f = branches.from_buses
    t = branches.to_buses
    nl = length(f)
    # Connection matrices
    Cf = sparse(1:nl, f, ones(nl), nl, nb)
    Ct = sparse(1:nl, t, ones(nl), nl, nb)

    Vf = Cf * V
    Vt = Ct * V

    Ev = sparse(1:nb, 1:nb, V./abs.(V), nb, nb)
    diagV = sparse(1:nb, 1:nb, V, nb, nb)
    diagVf = sparse(1:nl, 1:nl, Vf, nl, nl)
    diagVt = sparse(1:nl, 1:nl, Vt, nl, nl)

    dS_dVm = diagVf * Ct * conj(Ev) + conj(diagVt) * Cf * Ev
    dS_dVa = im * (-diagVf * Ct * conj(diagV) + conj(diagVt) * Cf * diagV)

    return (dS_dVm, dS_dVa)
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

