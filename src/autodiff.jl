
module AutoDiff

using SparseArrays

using CUDA
import ForwardDiff
import SparseDiffTools
using KernelAbstractions

using ..ExaPF: State, Control

import Base: show

"""
    AbstractJacobian

Automatic differentiation for the compressed Jacobian of
any nonlinear constraint ``h(x)``.

"""
abstract type AbstractJacobian end

"""
    AbstractHessian

Automatic differentiation for the adjoint-Hessian-vector product ``λ^⊤ H v`` of
any nonlinear constraint ``h(x)``.

"""
abstract type AbstractHessian end


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
    duals, @Const(x), @Const(v), @Const(map),
)
    i = @index(Global, Linear)

    duals[1, map[i]] = x[map[i]]
    duals[2, map[i]] = v[i]
end

function seed!(dest::AbstractVector{ForwardDiff.Dual{Nothing, T, 1}}, src, v, map, device) where {T}
    n = length(dest)
    dest_ = reshape(reinterpret(T, dest), 2, n)
    ndrange = length(map)
    ev = _seed_kernel!(device)(
        dest_, src, v, map, ndrange=ndrange, dependencies=Event(device))
    wait(ev)
end

function seed_coloring!(dest::AbstractVector{ForwardDiff.Dual{Nothing, T, N}}, coloring, map, device) where {T, N}
    n = length(dest)
    ncolors = N
    dest_ = reshape(reinterpret(T, dest), N+1, n)
    ndrange = (length(map), ncolors)
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

function getpartials_kernel!(hv::AbstractVector, adj_t1sx, map, device)
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

function partials_jac!(J, duals::AbstractVector{ForwardDiff.Dual{Nothing, T, N}}, coloring, device) where {T, N}
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

function partials_hess!(J, duals::AbstractVector{ForwardDiff.Dual{Nothing, T, N}}, map, coloring, device) where {T, N}
    n = length(duals)
    duals_ = reshape(reinterpret(T, duals), N+1, n)

    if isa(device, CPU)
        kernel! = partials_kernel_cpu!(device)
        ev = kernel!(J.colptr, J.rowval, J.nzval, duals_, map, coloring, ndrange=size(J,2), dependencies=Event(device))
    elseif isa(device, GPU)
        kernel! = partials_kernel_gpu!(device)
        ev = kernel!(J.rowPtr, J.colVal, J.nzVal, duals_, map, coloring, ndrange=size(J,1), dependencies=Event(device))
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

function set_value!(duals::AbstractVector{ForwardDiff.Dual{Nothing, T, N}}, primals::AbstractVector{T}, device) where {T,N}
    n = length(duals)
    duals_ = reshape(reinterpret(T, duals), N+1, n)
    ev = _set_value_kernel!(device)(
        duals_, primals, ndrange=n, dependencies=Event(device))
    wait(ev)
end

end
