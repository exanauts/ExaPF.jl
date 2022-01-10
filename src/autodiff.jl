
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
function _init_seed!(t1sseeds, t1sseedvecs, coloring, ncolor, nmap)
    for i in 1:nmap
        t1sseedvecs[:,i] .= 0
        @inbounds for j in 1:ncolor
            if coloring[i] == j
                t1sseedvecs[j,i] = 1.0
            end
        end
        t1sseeds[i] = ForwardDiff.Partials{ncolor, Float64}(NTuple{ncolor, Float64}(t1sseedvecs[:,i]))
    end
end

function init_seed(coloring, ncolor, nmap)
    t1sseeds = Vector{ForwardDiff.Partials{ncolor, Float64}}(undef, nmap)
    t1sseedvecs = zeros(Float64, ncolor, nmap)
    # The seeding is always done on the CPU since it's faster
    _init_seed!(t1sseeds, t1sseedvecs, Array(coloring), ncolor, nmap)
    return t1sseeds
end

@kernel function seed_kernel!(
    duals::AbstractArray{ForwardDiff.Dual{T, V, N}}, @Const(x),
    @Const(seeds)
) where {T,V,N}
    i = @index(Global, Linear)
    duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
end

"""
    seed!

Calling the seeding kernel.
Seeding is parallelized over the `ncolor` number of duals.

"""
function seed!(t1sseeds, varx, t1svarx, device)
    kernel! = seed_kernel!(device)
    ev = kernel!(t1svarx, varx, t1sseeds, ndrange=length(t1svarx), dependencies=Event(device))
    wait(ev)
end


# Get partials
@kernel function getpartials_jac_kernel!(compressedJ, @Const(duals))
    i, j = @index(Global, NTuple)
    compressedJ[j, i] = duals[j+1, i]
end

# Get partials for Hessian projection
@kernel function getpartials_hv_kernel!(hv, @Const(adj_t1sx), @Const(map))
    i = @index(Global, Linear)
    @inbounds begin
        hv[i] = ForwardDiff.partials(adj_t1sx[map[i]]).values[1]
    end
end

@kernel function getpartials_hess_kernel!(compressedH, @Const(duals), @Const(map))
    i, j = @index(Global, NTuple)
    compressedH[j, i] = duals[j+1, map[i]]
end

"""
    getpartials_kernel!(compressedJ, t1sF)

Calling the partial extraction kernel.
Extract the partials from the AutoDiff dual type on the target
device and put it in the compressed Jacobian `compressedJ`.

"""
function getpartials_kernel!(hv::AbstractVector, adj_t1sx, map, device)
    kernel! = getpartials_hv_kernel!(device)
    ev = kernel!(hv, adj_t1sx, map, ndrange=length(hv), dependencies=Event(device))
    wait(ev)
end

function partials_jac!(
    compressedJ::AbstractMatrix{T},
    duals::AbstractVector{ForwardDiff.Dual{Nothing, T, N}},
    device,
) where {T, N}
    n = length(duals)
    @assert size(compressedJ) == (N, n)
    duals_ = reshape(reinterpret(T, duals), N+1, n)
    ndrange = (n, N)
    ev = getpartials_jac_kernel!(device)(
        compressedJ, duals_,
        ndrange=ndrange, dependencies=Event(device),
    )
    wait(ev)
end

function partials_hess!(
    compressedH::AbstractMatrix,
    duals::AbstractVector{ForwardDiff.Dual{Nothing, T, N}},
    map, device,
) where {T, N}
    n = length(map)
    @assert size(compressedH) == (N, n)
    duals_ = reshape(reinterpret(Float64, duals), N+1, length(duals))
    ndrange = (n, N)
    ev = getpartials_hess_kernel!(device)(
        compressedH, duals_, map,
        ndrange=ndrange, dependencies=Event(device),
    )
    wait(ev)
end


# Uncompress kernels
@kernel function uncompress_kernel_gpu!(@Const(J_rowPtr), @Const(J_colVal), J_nzVal, @Const(compressedJ), @Const(coloring))
    i = @index(Global, Linear)
    @inbounds for j in J_rowPtr[i]:J_rowPtr[i+1]-1
        @inbounds J_nzVal[j] = compressedJ[coloring[J_colVal[j]], i]
    end
end

@kernel function uncompress_kernel_cpu!(J_colptr, J_rowval, J_nzval, compressedJ, coloring)
    # CSC is column oriented: nmap is equal to number of columns
    i = @index(Global, Linear)
    @inbounds for j in J_colptr[i]:J_colptr[i+1]-1
        @inbounds J_nzval[j] = compressedJ[coloring[i], J_rowval[j]]
    end
end

"""
    uncompress_kernel!(J, compressedJ, coloring)

Uncompress the compressed Jacobian matrix from `compressedJ`
to sparse CSC (on the CPU) or CSR (on the GPU).
"""
function uncompress_kernel!(J, compressedJ, coloring, device)
    if isa(device, CPU)
        kernel! = uncompress_kernel_cpu!(device)
        ev = kernel!(J.colptr, J.rowval, J.nzval, compressedJ, coloring, ndrange=size(J,2), dependencies=Event(device))
    elseif isa(device, GPU)
        kernel! = uncompress_kernel_gpu!(device)
        ev = kernel!(J.rowPtr, J.colVal, J.nzVal, compressedJ, coloring, ndrange=size(J,1), dependencies=Event(device))
    else
        error("Unknown device $device")
    end
    wait(ev)
end

end
