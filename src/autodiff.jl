
module AutoDiff

using SparseArrays

using CUDA
import ForwardDiff
import SparseDiffTools
using KernelAbstractions

using ..ExaPF: State, Control

import Base: show

"""
    AbstractStack{VT}

Abstract variable storing the inputs and the intermediate values in the expression tree.

"""
abstract type AbstractStack{VT} end

#=
    Generic expression
=#

"""
    AbstractExpression

Abstract type for differentiable function ``f(x)``.
Any `AbstractExpression` implements two functions: a forward
mode to evaluate ``f(x)``, and an adjoint to evaluate ``∂f(x)``.

### Forward mode
The direct evaluation of the function ``f`` is implemented as
```julia
(expr::AbstractExpression)(output::VT, stack::AbstractStack{VT}) where VT<:AbstractArray

```
the input being specified in `stack`, the results being stored in the array `output`.

### Reverse mode
The adjoint of the function is specified by the function `adjoint!`, with
the signature:
```julia
adjoint!(expr::AbstractExpression, ∂stack::AbstractStack{VT}, stack::AbstractStack{VT}, ̄v::VT) where VT<:AbstractArray

```
The variable `stack` stores the result of the direct evaluation, and is not
modified in `adjoint!`. The results are stored inside the adjoint stack
`∂stack`.

"""
abstract type AbstractExpression end

function (expr::AbstractExpression)(stack::AbstractStack)
    m = length(expr)
    output = similar(stack.input, m)
    expr(output, stack)
    return output
end

"""
    adjoint!(expr::AbstractExpression, ∂stack::AbstractStack{VT}, stack::AbstractStack{VT}, ̄v::VT) where VT<:AbstractArray

Compute the adjoint of the `AbstractExpression` `expr` with relation to the
adjoint vector `̄v`. The results are stored in the adjoint stack `∂stack`.
The variable `stack` stores the result of a previous direct evaluation, and is not
modified in `adjoint!`.

"""
function adjoint! end

"""
    AbstractJacobian

Automatic differentiation for the compressed Jacobian of
any nonlinear constraint ``h(x)``.

"""
abstract type AbstractJacobian end

function jacobian!(jac::AbstractJacobian, stack::AbstractStack)
    error("Mising method jacobian!(", typeof(jac), ", ", typeof(stack), ")")
end

"""
    AbstractHessianProd

Returns the adjoint-Hessian-vector product ``λ^⊤ H v`` of
any nonlinear constraint ``h(x)``.

"""
abstract type AbstractHessianProd end

"""
    AbstractHessianProd

Full sparse Hessian ``H`` of any nonlinear constraint ``h(x)``.

"""
abstract type AbstractFullHessian end

# Seeding

@kernel function _seed_coloring_kernel!(
    duals, @Const(coloring), @Const(map),
)
    i, j = @index(Global, NTuple)

    if coloring[i] == j
        duals[j+1, map[i]] = 1.0
    end
end

@kernel function _seed_kernel!(
    duals, @Const(v), @Const(map),
)
    i = @index(Global, Linear)

    duals[2, map[i]] = v[i]
end

"""
    seed!(
        H::AbstractHessianProd,
        v::AbstractVector{T},
    ) where {T}

Seed the duals with v to compute the Hessian vector product ``λ^⊤ H v``.

"""
function seed!(
    H::AbstractHessianProd,
    v::AbstractVector{T},
) where {T}
    dest = H.stack.input
    map = H.map
    device = H.model.device
    n = length(dest)
    dest_ = reshape(reinterpret(T, dest), 2, n)
    ndrange = length(map)
    ev = _seed_kernel!(device)(
        dest_, v, map, ndrange=ndrange, dependencies=Event(device))
    wait(ev)
end

"""
    seed_coloring!(
        M::Union{AbstractJacobian, AbstractFullHessian}
        coloring::AbstractVector,
    )

Seed the duals with the `coloring` based seeds to compute the Jacobian or Hessian ``M``.

"""
function seed_coloring!(
    M::Union{AbstractJacobian, AbstractFullHessian},
    coloring::AbstractVector,
)
    dest = M.stack.input
    _seed_coloring!(M, coloring, dest)
end

function _seed_coloring!(
    M::Union{AbstractJacobian, AbstractFullHessian},
    coloring::AbstractVector,
    dest::AbstractVector{ForwardDiff.Dual{Nothing, T, N}},
) where {T, N}
    dest = M.stack.input
    map = M.map
    device = M.model.device
    n = length(dest)
    dest_ = reshape(reinterpret(T, dest), N+1, n)
    ndrange = (length(map), N)
    ev = _seed_coloring_kernel!(device)(
        dest_, coloring, map, ndrange=ndrange, dependencies=Event(device))
    wait(ev)
end

# Get partials

# Get partials for Hessian projection
@kernel function getpartials_hv_kernel!(hv, @Const(adj_t1sx), @Const(map))
    i = @index(Global, Linear)
    @inbounds begin
        hv[i] = ForwardDiff.partials(adj_t1sx[map[i]]).values[1]
    end
end

"""
    getpartials_kernel!(hv::AbstractVector, H::AbstractHessianProd)

Extract partials from `ForwardDiff.Dual` numbers with only 1 partial when computing the Hessian vector product.

"""
function getpartials_kernel!(hv::AbstractVector, H::AbstractHessianProd)
    device = H.model.device
    map = H.map
    adj_t1sx = H.∂stack.input
    kernel! = getpartials_hv_kernel!(device)
    ev = kernel!(hv, adj_t1sx, map, ndrange=length(hv), dependencies=Event(device))
    wait(ev)
end

# Sparse Jacobian partials

@kernel function partials_kernel_gpu!(@Const(J_rowPtr), @Const(J_colVal), J_nzVal, @Const(duals), @Const(coloring))
    i = @index(Global, Linear)

    @inbounds for j in J_rowPtr[i]:J_rowPtr[i+1]-1
        @inbounds J_nzVal[j] = duals[coloring[J_colVal[j]]+1, i]
    end
end

@kernel function partials_kernel_cpu!(J_colptr, J_rowval, J_nzval, duals, coloring)
    # CSC is column oriented: nmap is equal to number of columns
    i = @index(Global, Linear)

    @inbounds for j in J_colptr[i]:J_colptr[i+1]-1
        @inbounds J_nzval[j] = duals[coloring[i]+1, J_rowval[j]]
    end
end

"""
    partials!(jac::AbstractJacobian)

Extract partials from Jacobian `jac` in `jac.J`.

"""
function partials!(jac::AbstractJacobian)
    J = jac.J
    N = jac.ncolors
    T = eltype(J)
    duals = jac.t1sF
    device = jac.model.device
    coloring = jac.coloring
    n = length(duals)

    duals_ = reshape(reinterpret(T, duals), N+1, n)

    if isa(device, CPU)
        kernel! = partials_kernel_cpu!(device)
        ev = kernel!(J.colptr, J.rowval, J.nzval, duals_, coloring, ndrange=size(J,2), dependencies=Event(device))
    elseif isa(device, GPU)
        kernel! = partials_kernel_gpu!(device)
        ev = kernel!(J.rowPtr, J.colVal, J.nzVal, duals_, coloring, ndrange=size(J,1), dependencies=Event(device))
    else
        error("Unknown device $device")
    end
    wait(ev)
end

# Sparse Hessian partials

@kernel function partials_kernel_gpu!(@Const(J_rowPtr), @Const(J_colVal), J_nzVal, @Const(duals), @Const(map), @Const(coloring))
    i = @index(Global, Linear)

    @inbounds for j in J_rowPtr[i]:J_rowPtr[i+1]-1
        @inbounds J_nzVal[j] = duals[coloring[J_colVal[j]]+1, map[i]]
    end
end

@kernel function partials_kernel_cpu!(J_colptr, J_rowval, J_nzval, duals, map, coloring)
    # CSC is column oriented: nmap is equal to number of columns
    i = @index(Global, Linear)

    @inbounds for j in J_colptr[i]:J_colptr[i+1]-1
        @inbounds J_nzval[j] = duals[coloring[i]+1, map[J_rowval[j]]]
    end
end

"""
    partials!(hess::AbstractFullHessian)

Extract partials from Hessian `hess` into `hess.H`.

"""
function partials!(hess::AbstractFullHessian)
    H = hess.H
    N = hess.ncolors
    T = eltype(H)
    duals = hess.∂stack.input
    device = hess.model.device
    coloring = hess.coloring
    map = hess.map
    n = length(duals)

    duals_ = reshape(reinterpret(T, duals), N+1, n)

    if isa(device, CPU)
        kernel! = partials_kernel_cpu!(device)
        ev = kernel!(H.colptr, H.rowval, H.nzval, duals_, map, coloring, ndrange=size(H,2), dependencies=Event(device))
    elseif isa(device, GPU)
        kernel! = partials_kernel_gpu!(device)
        ev = kernel!(H.rowPtr, H.colVal, H.nzVal, duals_, map, coloring, ndrange=size(H,1), dependencies=Event(device))
    else
        error("Unknown device $device")
    end
    wait(ev)
end

@kernel function _set_value_kernel!(
    duals, @Const(primals),
)
    i = @index(Global, Linear)

    duals[1, i] = primals[i]
end

"""
    set_value!(
        jac,
        primals::AbstractVector{T}
    ) where {T}

Set values of `ForwardDiff.Dual` numbers in `jac` to `primals`.

"""
function set_value!(
    jac,
    primals::AbstractVector{T}
) where {T}
    duals = jac.stack.input
    device = jac.model.device
    n = length(duals)
    N = jac.ncolors
    duals_ = reshape(reinterpret(T, duals), N+1, n)
    ev = _set_value_kernel!(device)(
        duals_, primals, ndrange=n, dependencies=Event(device))
    wait(ev)
end

end
