
# By default, generic Julia functions are not considered as constraint:
is_constraint(::Function) = false

# Is the function linear in the polar formulation?
is_linear(polar::PolarForm, ::Function) = false

struct FullSpaceJacobian{Jac}
    x::Jac
    u::Jac
end

struct FullSpaceHessian{SpMT}
    xx::SpMT
    xu::SpMT
    uu::SpMT
end


include("power_balance.jl")
include("voltage_magnitude.jl")
include("active_power.jl")
include("reactive_power.jl")
include("line_flow.jl")

# Generic functions

## Adjoint
# TODO: port to new stack
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
function jtprod!(
    polar::PolarForm,
    func::Function,
    stack::AdjointPolar,
    buffer::PolarNetworkState,
    v::AbstractVector,
)
    @assert is_constraint(func)

    m = size_constraint(polar, func)
    # Adjoint w.r.t. vm, va, pinj, qinj
    fill!(stack.∂vm, 0)
    fill!(stack.∂va, 0)
    fill!(stack.∂pinj, 0.0)
    fill!(stack.∂qinj, 0.0)
    cons = buffer.balance ; fill!(cons, 0.0)
    adjoint!(
        polar, func,
        cons, v,
        buffer.vmag, stack.∂vm,
        buffer.vang, stack.∂va,
        buffer.pinj, stack.∂pinj,
        buffer.qinj, stack.∂qinj,
    )

    ∂x = stack.∂x ; fill!(∂x, 0.0)
    ∂u = stack.∂u ; fill!(∂u, 0.0)
    adjoint_transfer!(
        polar,
        ∂u, ∂x,
        stack.∂vm, stack.∂va, stack.∂pinj, stack.∂qinj,
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

# Utilities for AutoDiff
function _build_jacobian(polar::PolarForm, cons::Function, X::Union{State, Control})
    if is_linear(polar, cons)
        return AutoDiff.ConstantJacobian(polar, cons, X)
    else
        return AutoDiff.Jacobian(polar, cons, X)
    end
end

_build_hessian(polar::PolarForm, cons::Function) = AutoDiff.Hessian(polar, cons)

function JacobianStorage(
    polar::PolarForm{T, VI, VT, MT},
    cons::Function,
) where {T, VI, VT, MT}
    @assert is_constraint(cons)
    Jx = _build_jacobian(polar, cons, State())
    Ju = _build_jacobian(polar, cons, Control())
    return JacobianStorage(Jx, Ju)
end

