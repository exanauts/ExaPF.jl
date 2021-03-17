
# By default, generic Julia functions are not considered as constraint:
is_constraint(::Function) = false

# Is the function linear in the polar formulation?
is_linear(polar::PolarForm, ::Function) = false


include("power_balance.jl")
include("voltage_magnitude.jl")
include("active_power.jl")
include("reactive_power.jl")
include("line_flow.jl")

# By default, function does not have any intermediate state
_get_intermediate_stack(polar::PolarForm, func::Function, VT) = nothing

function _get_intermediate_stack(
    polar::PolarForm, func::F, VT
) where {F <: Union{typeof(reactive_power_constraints), typeof(flow_constraints), typeof(power_balance)}}
    nlines = PS.get(polar.network, PS.NumberOfLines())
    # Take care that flow_constraints needs a buffer with a different size
    nnz = isa(func, typeof(flow_constraints)) ? nlines : length(polar.topology.ybus_im.nzval)
    # Return a NamedTuple storing all the intermediate states
    return (
        ∂edge_vm_fr = xzeros(VT, nnz),
        ∂edge_va_fr = xzeros(VT, nnz),
        ∂edge_vm_to = xzeros(VT, nnz),
        ∂edge_va_to = xzeros(VT, nnz),
    )
end

# Generic functions
function AutoDiff.PullbackMemory(
    polar::PolarForm, func::Function, VT; with_stack=true,
)
    @assert is_constraint(func)
    stack = (with_stack) ? AdjointPolar(polar) : nothing
    intermediate = _get_intermediate_stack(polar, func, VT)
    return AutoDiff.PullbackMemory(func, stack, intermediate)
end

## Adjoint
function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.PullbackMemory,
    ∂cons, cons,
    buffer,
)
    stack = pbm.stack
    reset!(stack)
    adjoint!(
        polar, pbm,
        cons, ∂cons,
        buffer.vmag, stack.∂vm,
        buffer.vang, stack.∂va,
        buffer.pinj, stack.∂pinj,
    )
end

## Jacobian-transpose vector product
function jtprod!(
    polar::PolarForm,
    pbm::AutoDiff.PullbackMemory,
    buffer::PolarNetworkState,
    v::AbstractVector,
)
    # Adjoint w.r.t. vm, va, pinj, qinj
    stack = pbm.stack
    reset!(stack)
    cons = buffer.balance ; fill!(cons, 0.0) # TODO
    adjoint!(
        polar, pbm,
        cons, v,
        buffer.vmag, stack.∂vm,
        buffer.vang, stack.∂va,
        buffer.pinj, stack.∂pinj,
    )
    adjoint_transfer!(
        polar,
        stack.∂u, stack.∂x,
        stack.∂vm, stack.∂va, stack.∂pinj,
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

function FullSpaceJacobian(
    polar::PolarForm{T, VI, VT, MT},
    cons::Function,
) where {T, VI, VT, MT}
    @assert is_constraint(cons)
    Jx = _build_jacobian(polar, cons, State())
    Ju = _build_jacobian(polar, cons, Control())
    return FullSpaceJacobian(Jx, Ju)
end

