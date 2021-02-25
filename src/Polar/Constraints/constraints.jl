
# By default, generic Julia functions are not considered as constraint:
is_constraint(::Function) = false

struct FullSpaceJacobian{Jac}
    x::Jac
    u::Jac
end


include("power_balance.jl")
include("voltage_magnitude.jl")
include("active_power.jl")
include("reactive_power.jl")
include("line_flow.jl")

function jacobian_sparsity(polar::PolarForm, func::Function, xx::AbstractVariable)
    nbus = get(polar, PS.NumberOfBuses())
    Vre = Float64[i for i in 1:nbus]
    Vim = Float64[i for i in nbus+1:2*nbus]
    V = Vre .+ im .* Vim
    return matpower_jacobian(polar, xx, func, V)
end

# MATPOWER's Jacobians
Base.@deprecate(residual_jacobian, matpower_jacobian)

# Jacobian Jₓ (from Matpower)
"""
    residual_jacobian(V, Ybus, pv, pq)

Compute the Jacobian w.r.t. the state `x` of the power
balance function [`power_balance`](@ref).

# Note
Code adapted from MATPOWER.
"""
function residual_jacobian(::State, V, Ybus, pv, pq, ref)
    # error("deprecated")
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
    j12 = real(dSbus_dVm[[pv; pq], pq])
    j21 = imag(dSbus_dVa[pq, [pv; pq]])
    j22 = imag(dSbus_dVm[pq, pq])

    J = [j11 j12; j21 j22]
end


# Jacobian Jᵤ (from Matpower)
function residual_jacobian(::Control, V, Ybus, pv, pq, ref)
    # error("deprecated")
    dSbus_dVm, _ = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVm[[pv; pq], [ref; pv; pv]])
    j21 = imag(dSbus_dVm[pq, [ref; pv; pv]])
    J = [j11; j21]
end


# Jacobian wrt active power generation
function active_power_jacobian(::State, V, Ybus, pv, pq, ref)
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVa[ref, [pv; pq]])
    j12 = real(dSbus_dVm[ref, pq])
    J = [
        j11 j12
        spzeros(length(pv), length(pv) + 2 * length(pq))
    ]
end

function active_power_jacobian(::Control, V, Ybus, pv, pq, ref)
    ngen = length(pv) + length(ref)
    npv = length(pv)
    dSbus_dVm, _ = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVm[ref, [ref; pv]])
    j12 = sparse(I, npv, npv)
    return [
        j11 spzeros(length(ref), npv)
        spzeros(npv, ngen) j12
    ]
end

function active_power_jacobian(
    polar::PolarForm,
    r::AbstractVariable,
    buffer::PolarNetworkState,
)
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    V = buffer.vmag .* exp.(im .* buffer.vang)
    return active_power_jacobian(r, V, polar.network.Ybus, pv, pq, ref)
end
