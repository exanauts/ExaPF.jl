
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

# Generic functions
## Adjoint
function adjoint!(
    polar::PolarForm,
    func::Function,
    ∂cons, cons,
    stack, buffer,
)
    @assert is_constraint(func)
    ∂pinj = similar(buffer.vmag) ; fill!(∂pinj, 0.0)
    ∂qinj = nothing # TODO
    fill!(stack.∂vm, 0)
    fill!(stack.∂va, 0)
    adjoint!(
        polar, func,
        cons, ∂cons,
        buffer.vmag, stack.∂vm,
        buffer.vang, stack.∂va,
        buffer.pinj, ∂pinj,
        buffer.qinj, ∂qinj,
    )
end

## Jacobian-transpose vector product
function jtprod(
    polar::PolarForm,
    func::Function,
    stack,
    buffer,
    v::AbstractVector,
)
    @assert is_constraint(func)

    m = size_constraint(polar, func)
    # Adjoint w.r.t. vm, va, pinj, qinj
    ∂pinj = similar(buffer.vmag) ; fill!(∂pinj, 0.0)
    ∂qinj = nothing # TODO
    fill!(stack.∂vm, 0)
    fill!(stack.∂va, 0)
    cons = similar(buffer.vmag, m) ; fill!(cons, 0.0)
    adjoint!(
        polar, func,
        cons, v,
        buffer.vmag, stack.∂vm,
        buffer.vang, stack.∂va,
        buffer.pinj, ∂pinj,
        buffer.qinj, ∂qinj,
    )

    # Adjoint w.r.t. x and u
    ∂x = stack.∇fₓ ; fill!(∂x, 0.0)
    ∂u = stack.∇fᵤ ; fill!(∂u, 0.0)
    adjoint_transfer!(
        polar,
        ∂u, ∂x,
        stack.∂vm, stack.∂va, ∂pinj, ∂qinj,
    )
end

## Sparsity detection
function jacobian_sparsity(polar::PolarForm, func::Function, xx::AbstractVariable)
    nbus = get(polar, PS.NumberOfBuses())
    Vre = Float64[i for i in 1:nbus]
    Vim = Float64[i for i in nbus+1:2*nbus]
    V = Vre .+ im .* Vim
    return matpower_jacobian(polar, xx, func, V)
end

function matpower_jacobian(polar::PolarForm, func::Function, X::AbstractVariable, buffer::PolarNetworkState)
    V = buffer.vmag .* exp.(im .* buffer.vang)
    return matpower_jacobian(polar, X, func, V)
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

