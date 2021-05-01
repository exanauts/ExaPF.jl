
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

abstract type AbstractAdjointStack{VT} end

"""
    jacobian!(form::AbstractFormulation, jac::AutoDiff.AbstractJacobian, x)

Update inplace the Jacobian ``J`` stored inside `jac` at a new point `x`.
"""
function jacobian! end

"""
    adj_hessian_prod!(form::AbstractFormulation, H::AutoDiff.AbstractHessian, hv, x, λ, v)

Compute the adjoint-Hessian-vector product ``λ^⊤ H v`` at a given
point `x`, and store the result inplace in vector `hv`.

"""
function adj_hessian_prod! end

"""
    AutoDiff.Jacobian <: AbstractJacobian

Creates an object to compute the Jacobian with ForwardDiff.

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

"""
    AutoDiff.ConstantJacobian <: AbstractJacobian

Creates a constant Jacobian object for a linear function ``h(x)``.
Using a `ConstantJacobian` object allows to avoid computing
the full Jacobian with AutoDiff when it is not necessary.
"""
struct ConstantJacobian{SMT} <: AbstractJacobian
    J::SMT
end

"""
    AutoDiff.Hessian

Creates an object for computing Hessian adjoint tangent projections.

* `func::Func`: base function to differentiate.
* `host_t1sseeds::VHP`: Seeding vector for seeding on the host.
* `t1sseeds::VP`: The seeding vector for AD built based on the coloring.
* `x::VT`: Input array of passive type. This includes both state and control.
* `t1sF::VD`: Output array of active (AD) type.
* `∂t1sF::VD`: Adjoint of the output array.
* `t1sx::VD`: Input array of active type.
* `∂t1sx::VD`: Adjoint of the input array.
* `map::VI`: State and control mapping to array `x`
* `varx::SubT`: View of `map` on `x`
* `t1svarx::SubD`: Active (AD) view of `map` on `x`
* `buffer::Buff`: cache for computing the adjoint (could be `Nothing`)
"""
struct Hessian{Func, VI, VT, VHP, VP, VD, SubT, SubD, Buff} <: AbstractHessian
    func::Func
    host_t1sseeds::VHP # Needed because seeds have to be created on the host
    t1sseeds::VP
    x::VT
    t1sF::VD
    ∂t1sF::VD
    t1sx::VD
    ∂t1sx::VD
    map::VI
    varx::SubT
    t1svarx::SubD
    buffer::Buff
end

# Cache for adjoint
"""
    TapeMemory{F, S, I}

This object is used as a buffer to compute the adjoint of a given function
``h(x)``. It stores internally all intermediate values necessary
to compute the adjoint, and cache the stack used in the backward pass.

## Note
This structure is largely inspired from [ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/stable/design/changing_the_primal.html#The-Journey-to-rrule).
"""
struct TapeMemory{F, S, I}
    func::F
    stack::S
    intermediate::I
end

# Seeding
# TODO
@kernel function _init_seed!(t1sseeds, t1sseedvecs, @Const(coloring), ncolor)
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
    ev = _init_seed!(CPU())(t1sseeds, t1sseedvecs, Array(coloring), ncolor, ndrange=nmap, dependencies=Event(CPU()))
    wait(ev)
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
@kernel function getpartials_kernel!(compressedJ, @Const(t1sF))
    i = @index(Global, Linear)
    for j in eachindex(ForwardDiff.partials.(t1sF[i]).values)
        @inbounds compressedJ[j, i] = ForwardDiff.partials.(t1sF[i]).values[j]
    end
end

# Get partials for Hessian projection
@kernel function getpartials_hv_kernel!(hv, @Const(adj_t1sx), @Const(map))
    i = @index(Global, Linear)
    hv[i] = ForwardDiff.partials(adj_t1sx[map[i]]).values[1]
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

function getpartials_kernel!(compressedJ::AbstractMatrix, t1sF, device)
    kernel! = getpartials_kernel!(device)
    ev = kernel!(compressedJ, t1sF, ndrange=length(t1sF), dependencies=Event(device))
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

function Base.show(io::IO, jacobian::Jacobian)
    println(io, "A AutoDiff Jacobian for $(jacobian.func) (w.r.t. $(jacobian.var))")
    ncolor = size(jacobian.compressedJ, 1)
    print(io, "Number of Jacobian colors: ", ncolor)
end

end
