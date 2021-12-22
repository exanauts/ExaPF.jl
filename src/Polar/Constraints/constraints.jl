
# By default, generic Julia functions are not considered as constraint:
is_constraint(::Function) = false

# Is the function linear in the polar formulation?
is_linear(polar::PolarForm, ::Function) = false


include("power_balance.jl")
include("power_injection.jl")
include("voltage_magnitude.jl")
include("active_power.jl")
include("reactive_power.jl")
include("line_flow.jl")
include("ramping_rate.jl")
include("network_operation.jl")
include("basis.jl")

# By default, function does not have any intermediate state
_get_intermediate_stack(polar::PolarForm, func::Function, VT, nbatch) = nothing

function _get_intermediate_stack(
    polar::PolarForm, func::F, VT, nbatch
) where {F <: Union{typeof(reactive_power_constraints), typeof(flow_constraints), typeof(power_balance), typeof(bus_power_injection), typeof(network_basis)}}
    nlines = PS.get(polar.network, PS.NumberOfLines())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    # Take care that flow_constraints needs a buffer with a different size
    nnz = if isa(func, typeof(flow_constraints))  || isa(func, typeof(network_basis))
        nlines
    else
        length(polar.topology.ybus_im.nzval)
    end

    Cf = nothing
    Ct = nothing
    if isa(func, typeof(network_basis)) || isa(func, typeof(flow_constraints))
        SMT, _ = get_jacobian_types(polar.device)
        Cf = sparse(polar.network.lines.from_buses, 1:nlines, ones(nlines), nbus, nlines) |> SMT
        Ct = sparse(polar.network.lines.to_buses, 1:nlines, ones(nlines), nbus, nlines) |> SMT
    end

    # Return a NamedTuple storing all the intermediate states
    if nbatch == 1
        return (
            Cf=Cf, Ct=Ct,
            ∂edge_vm_fr = VT(undef, nnz),
            ∂edge_va_fr = VT(undef, nnz),
            ∂edge_vm_to = VT(undef, nnz),
            ∂edge_va_to = VT(undef, nnz),
        )
    else
        return (
            Cf=Cf, Ct=Ct,
            ∂edge_vm_fr = VT(undef, nnz, nbatch),
            ∂edge_va_fr = VT(undef, nnz, nbatch),
            ∂edge_vm_to = VT(undef, nnz, nbatch),
            ∂edge_va_to = VT(undef, nnz, nbatch),
        )
    end
end

# Generic functions
function AutoDiff.TapeMemory(
    polar::PolarForm, func::Function, VT; with_stack=true, nbatch=1,
)
    @assert is_constraint(func)
    stack = (with_stack) ? AdjointPolar(polar) : nothing
    intermediate = _get_intermediate_stack(polar, func, VT, nbatch)
    return AutoDiff.TapeMemory(func, stack, intermediate)
end

## Adjoint
function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory,
    ∂cons, cons, buffer,
)
    stack = pbm.stack
    reset!(stack)
    adjoint!(
        polar, pbm,
        cons, ∂cons,
        buffer.vmag, stack.∂vm,
        buffer.vang, stack.∂va,
        buffer.pnet, stack.∂pinj,
        buffer.pload, buffer.qload,
    )
end

## Jacobian-transpose vector product
function jacobian_transpose_product!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory,
    buffer::PolarNetworkState,
    v::AbstractVector,
)
    stack = pbm.stack
    reset!(stack)
    cons = buffer.balance ; fill!(cons, 0.0) # TODO
    adjoint!(
        polar, pbm,
        cons, v,
        buffer.vmag, stack.∂vm,
        buffer.vang, stack.∂va,
        buffer.pnet, stack.∂pinj,
        buffer.pload, buffer.qload,
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
    V = voltage_host(buffer)
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

function FullSpaceJacobian(
    polar::PolarForm{T, VI, VT, MT},
    cons::Function,
) where {T, VI, VT, MT}
    @assert is_constraint(cons)
    Jx = _build_jacobian(polar, cons, State())
    Ju = _build_jacobian(polar, cons, Control())
    return FullSpaceJacobian(Jx, Ju)
end

