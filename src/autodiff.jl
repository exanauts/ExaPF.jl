
module AutoDiff

using SparseArrays

using CUDA
import CUDA.CUSPARSE
import ForwardDiff
import SparseDiffTools
using KernelAbstractions

using ..ExaPF: Spmat, xzeros, State, Control

import Base: show

"""
    AbstractJacobian

Automatic differentiation for the compressed Jacobians of the
constraints `g(x,u)` with respect to the state `x` and the control `u`
(here called design).

TODO: Use dispatch to unify the code of the state and control Jacobian.
This is currently not done because the abstraction of the indexing is not yet resolved.

"""
abstract type AbstractJacobian end
abstract type AbstractHessian end

abstract type AbstractAdjointStack end

function jacobian! end

function adj_hessian_prod! end

"""
    AutoDiff.Jacobian

Creates an object for the Jacobian.

### Attributes

* `func::Func`: base function to differentiate
* `var::Union{State,Control}`: specify whether we are differentiating w.r.t. the state or the control.
* `J::SMT`: Sparse uncompressed Jacobian to be used by linear solver. This is either of type `SparseMatrixCSC` or `CuSparseMatrixCSR`.
* `compressedJ::MT`: Dense compressed Jacobian used for updating values through AD either of type `Matrix` or `CuMatrix`.
* `coloring::VI`: Row coloring of the Jacobian.
* `t1sseeds::VP`: The seeding vector for AD built based on the coloring.
* `t1sF::VD`: Output array of active (AD) type.
* `x::VT`: Input array of passive type. This includes both state and control.
* `t1sx::VD`: Input array of active type.
* `map::VI`: State and control mapping to array `x`
* `varx::SubT`: View of `map` on `x`
* `t1svarx::SubD`: Active (AD) view of `map` on `x`

"""
struct Jacobian{Func, VI, VT, MT, SMT, VP, VD, SubT, SubD} <: AbstractJacobian
    func::Func
    var::Union{State,Control}
    J::SMT
    compressedJ::MT
    coloring::VI
    t1sseeds::VP
    t1sF::VD
    x::VT
    t1sx::VD
    map::VI
    # Cache views on x and its dual vector to avoid reallocating on the GPU
    varx::SubT
    t1svarx::SubD
end

struct ConstantJacobian{SMT} <: AbstractJacobian
    J::SMT
end

"""
    AutoDiff.Hessian

Creates an object for computing Hessian adjoint tangent projections.

* `t1sseeds::VP`: The seeding vector for AD built based on the coloring.
* `t1sF::VD`: Output array of active (AD) type.
* `x::VT`: Input array of passive type. This includes both state and control.
* `t1sx::VD`: Input array of active type.
* `map::VI`: State and control mapping to array `x`
* `varx::SubT`: View of `map` on `x`
* `t1svarx::SubD`: Active (AD) view of `map` on `x`
"""
struct Hessian{Func, VI, VT, MT, SMT, VP, VD, SubT, SubD} <: AbstractHessian
    func::Func
    t1sseeds::VP
    t1sF::VD
    ∂t1sF::VD
    x::VT
    t1sx::VD
    ∂t1sx::VD
    map::VI
    varx::SubT
    t1svarx::SubD
end


# Seeding
@kernel function _init_seed!(t1sseeds, t1sseedvecs, coloring, ncolor)
    i = @index(Global, Linear)
    t1sseedvecs[:,i] .= 0
    @inbounds for j in 1:ncolor
        if coloring[i] == j
            t1sseedvecs[j,i] = 1.0
        end
    end
    t1sseeds[i] = ForwardDiff.Partials{ncolor, Float64}(NTuple{ncolor, Float64}(t1sseedvecs[:,i]))
end

function init_seed(coloring, ncolor, nmap)
    t1sseeds = Vector{ForwardDiff.Partials{ncolor, Float64}}(undef, nmap)
    t1sseedvecs = zeros(Float64, ncolor, nmap)
    # The seeding is always done on the CPU since it's faster
    ev = _init_seed!(CPU())(t1sseeds, t1sseedvecs, Array(coloring), ncolor, ndrange=nmap)
    wait(ev)
    return t1sseeds
end

@kernel function seed_kernel!(
    duals::AbstractArray{ForwardDiff.Dual{T, V, N}}, x,
    seeds::AbstractArray{ForwardDiff.Partials{N, V}}
) where {T,V,N}
    i = @index(Global, Linear)
    duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
end

"""
    seed!

Calling the seeding kernel.
Seeding is parallelized over the `ncolor` number of duals.

"""
function seed!(t1sseeds, varx, t1svarx, nbus) where {N, V}
    if isa(t1sseeds, Vector)
        kernel! = seed_kernel!(CPU())
    else
        kernel! = seed_kernel!(CUDADevice())
    end
    ev = kernel!(t1svarx, varx, t1sseeds, ndrange=length(t1svarx))
    wait(ev)
end


# Get partials
@kernel function getpartials_kernel_cpu!(compressedJ, t1sF)
    i = @index(Global, Linear)
    compressedJ[:, i] .= ForwardDiff.partials.(t1sF[i]).values
end

@kernel function getpartials_kernel_gpu!(compressedJ, t1sF)
    i = @index(Global, Linear)
    for j in eachindex(ForwardDiff.partials.(t1sF[i]).values)
        @inbounds compressedJ[j, i] = ForwardDiff.partials.(t1sF[i]).values[j]
    end
end

"""
    getpartials_kernel!(compressedJ, t1sF, nbus)

Calling the partial extraction kernel.
Extract the partials from the AutoDiff dual type on the target
device and put it in the compressed Jacobian `compressedJ`.

"""
function getpartials_kernel!(compressedJ, t1sF, nbus)
    if isa(compressedJ, Array)
        kernel! = getpartials_kernel_cpu!(CPU())
    else
        kernel! = getpartials_kernel_gpu!(CUDADevice())
    end
    ev = kernel!(compressedJ, t1sF, ndrange=length(t1sF))
    wait(ev)
end


# Uncompress kernels
@kernel function uncompress_kernel_gpu!(J_rowPtr, J_colVal, J_nzVal, compressedJ, coloring)
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
function uncompress_kernel!(J, compressedJ, coloring)
    if isa(J, SparseArrays.SparseMatrixCSC)
        kernel! = uncompress_kernel_cpu!(CPU())
        ev = kernel!(J.colptr, J.rowval, J.nzval, compressedJ, coloring, ndrange=size(J,2))
    else
        kernel! = uncompress_kernel_gpu!(CUDADevice())
        ev = kernel!(J.rowPtr, J.colVal, J.nzVal, compressedJ, coloring, ndrange=size(J,1))
    end
    wait(ev)
end

"""
    residual_hessian_adj_tgt!(H::Hessian,
                        residual_adj_polar!,
                        lambda, tgt,
                        vm, va, ybus_re, ybus_im, pinj, qinj, pv, pq, ref,
                        nbus)

Update the sparse Jacobian entries using AutoDiff. No allocations are taking place in this function.

* `H::Hessian`: Factory created Jacobian object to update
* `residual_adj_polar`: Adjoint function of residual
* `lambda`: Input adjoint, usually lambda from a Langrangian
* `tgt`: A tangent direction or vector that the Hessian should be multiplied with
* `vm, va, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus`: Inputs both
  active and passive parameters. Active inputs are mapped to `x` via the preallocated views.
"""
function tgt_adj_residual_hessian!(H::Hessian,
                             adj_residual_polar!,
                             lambda, tgt,
                             v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus)
    @warn("Function `tgt_adj_residual_hessian!` is deprecated.")
    x = H.x
    ntgt = length(tgt)
    nvbus = length(v_m)
    ninj = length(pinj)
    t1sx = H.t1sx
    adj_t1sx = similar(t1sx)
    t1sF = H.t1sF
    adj_t1sF = similar(t1sF)
    x[1:nvbus] .= v_m
    x[nvbus+1:2*nvbus] .= v_a
    x[2*nvbus+1:2*nvbus+ninj] .= pinj
    t1sx .= H.x
    adj_t1sx .= 0.0
    t1sF .= 0.0
    adj_t1sF .= lambda
    # Seeding
    nmap = length(H.map)
    t1sseedvec = zeros(Float64, length(x))
    for i in 1:nmap
        H.t1sseeds[i] = ForwardDiff.Partials{1, Float64}(NTuple{1, Float64}(tgt[i]))
    end
    seed!(H.t1sseeds, H.varx, H.t1svarx, nbus)
    adj_residual_polar!(
        t1sF, adj_t1sF,
        view(t1sx, 1:nvbus), view(adj_t1sx, 1:nvbus),
        view(t1sx, nvbus+1:2*nvbus), view(adj_t1sx, nvbus+1:2*nvbus),
        ybus_re, ybus_im,
        view(t1sx, 2*nvbus+1:2*nvbus+ninj), view(adj_t1sx, 2*nvbus+1:2*nvbus+ninj),
        qinj,
        pv, pq, nbus
    )
    # TODO, this is redundant
    ps = ForwardDiff.partials.(adj_t1sx[H.map])
    res = similar(tgt)
    res .= 0.0
    for i in 1:length(ps)
        res[i] = ps[i].values[1]
    end
    return res
end

function Base.show(io::IO, jacobian::Jacobian)
    println(io, "A AutoDiff Jacobian for $(jacobian.func) (w.r.t. $(jacobian.var))")
    ncolor = size(jacobian.compressedJ, 1)
    print(io, "Number of Jacobian colors: ", ncolor)
end

end
