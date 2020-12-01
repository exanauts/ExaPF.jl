function _matpower_residual_jacobian(V, Ybus)
    n = size(V, 1)
    Ibus = Ybus*V
    diagV       = sparse(1:n, 1:n, V, n, n)
    diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
    diagVnorm   = sparse(1:n, 1:n, V./abs.(V), n, n)

    dSbus_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
    dSbus_dVa = 1im * diagV * conj(diagIbus - Ybus * diagV)
    return (dSbus_dVm, dSbus_dVa)
end

# Jacobian Jₓ (from Matpower)
"""
    residual_jacobian(V, Ybus, pv, pq)

Compute the Jacobian w.r.t. the state `x` of the power
balance function [`power_balance`](@ref).

# Note
Code adapted from MATPOWER.
"""
function residual_jacobian(::State, V, Ybus, pv, pq, ref)
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
    j12 = real(dSbus_dVm[[pv; pq], pq])
    j21 = imag(dSbus_dVa[pq, [pv; pq]])
    j22 = imag(dSbus_dVm[pq, pq])

    J = [j11 j12; j21 j22]
end

# Jacobian Jᵤ (from Matpower)
function residual_jacobian(::Control, V, Ybus, pv, pq, ref)
    dSbus_dVm, _ = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVm[[pv; pq], [ref; pv; pv]])
    j21 = imag(dSbus_dVm[pq, [ref; pv; pv]])
    J = [j11; j21]
end

function residual_jacobian(A::Attr, polar::PolarForm) where {Attr <: Union{State, Control}}
    pf = polar.network
    ref = polar.network.ref
    pv = polar.network.pv
    pq = polar.network.pq
    n = PS.get(pf, PS.NumberOfBuses())

    Y = pf.Ybus
    # Randomized inputs
    Vre = rand(n)
    Vim = rand(n)
    V = Vre .+ 1im .* Vim
    return residual_jacobian(A, V, Y, pv, pq, ref)
end
_sparsity_pattern(polar::PolarForm) = findnz(residual_jacobian(State(), polar))

