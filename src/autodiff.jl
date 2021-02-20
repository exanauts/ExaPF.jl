
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
struct StateJacobian <: AbstractJacobian end
struct ControlJacobian <: AbstractJacobian end
abstract type AbstractHessian end

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

"""
    Jacobian

Creates an object for the Jacobian

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
struct Jacobian{VI, VT, MT, SMT, VP, VD, SubT, SubD}
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
    function Jacobian(structure, F, vm, va, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, type)
        nvbus = length(vm)
        npbus = length(pinj)
        nref = length(ref)
        if F isa Array
            VI = Vector{Int}
            VT = Vector{Float64}
            MT = Matrix{Float64}
            SMT = SparseMatrixCSC{Float64,Int}
            A = Vector
        elseif F isa CUDA.CuArray
            VI = CUDA.CuVector{Int}
            VT = CUDA.CuVector{Float64}
            MT = CUDA.CuMatrix{Float64}
            SMT = CUSPARSE.CuSparseMatrixCSR{Float64}
            A = CUDA.CuVector
        else
            error("Wrong array type ", typeof(F))
        end

        map = VI(structure.map)
        nmap = length(structure.map)
        hybus_re = Spmat{Vector{Int}, Vector{Float64}}(ybus_re)
        hybus_im = Spmat{Vector{Int}, Vector{Float64}}(ybus_im)
        Yre = SparseMatrixCSC{Float64,Int64}(nvbus, nvbus, hybus_re.colptr, hybus_re.rowval, hybus_re.nzval)
        Yim = SparseMatrixCSC{Float64,Int64}(nvbus, nvbus, hybus_im.colptr, hybus_im.rowval, hybus_im.nzval)
        Y = Yre .+ 1im .* Yim
        Vre = Float64.([i for i in 1:nvbus])
        Vim = Float64.([i for i in nvbus+1:2*nvbus])
        V = Vre .+ 1im .* Vim
        if isa(type, StateJacobian)
            variable = State()
            x = VT(zeros(Float64, 2*nvbus))
        elseif isa(type, ControlJacobian)
            variable = Control()
            x = VT(zeros(Float64, npbus + nvbus))
        else
            error("Unsupported Jacobian type. Must be either ControlJacobian or StateJacobian.")
        end
        J = structure.sparsity(variable, V, Y, pv, pq, ref)
        coloring = VI(SparseDiffTools.matrix_colors(J))
        ncolor = size(unique(coloring),1)
        if F isa CUDA.CuArray
            J = CUSPARSE.CuSparseMatrixCSR(J)
        end
        t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
        # The seeding is always done on the CPU since it's faster
        init_seed_kernel! = _init_seed!(CPU())
        t1sx = A{t1s{ncolor}}(x)
        t1sF = A{t1s{ncolor}}(zeros(Float64, length(F)))
        t1sseeds = Array{ForwardDiff.Partials{ncolor,Float64}}(undef, nmap)
        # Do the seeding on the CPU
        t1sseedvecs = zeros(Float64, ncolor, nmap)
        ev = init_seed_kernel!(t1sseeds, t1sseedvecs, Array(coloring), ncolor, ndrange=nmap)
        wait(ev)
        # Move the seeds over to the GPU
        gput1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(t1sseeds)
        compressedJ = MT(zeros(Float64, ncolor, length(F)))
        varx = view(x, map)
        t1svarx = view(t1sx, map)
        return new{VI, VT, MT, SMT, typeof(gput1sseeds), typeof(t1sx), typeof(varx), typeof(t1svarx)}(
            J, compressedJ, coloring, gput1sseeds, t1sF, x, t1sx, map, varx, t1svarx
        )
    end
end

"""
    Hessian

Creates an object for computing Hessian adjoint tangent projections

* `t1sseeds::VP`: The seeding vector for AD built based on the coloring.
* `t1sF::VD`: Output array of active (AD) type.
* `x::VT`: Input array of passive type. This includes both state and control.
* `t1sx::VD`: Input array of active type.
* `map::VI`: State and control mapping to array `x`
* `varx::SubT`: View of `map` on `x`
* `t1svarx::SubD`: Active (AD) view of `map` on `x`
"""
struct Hessian{VI, VT, MT, SMT, VP, VD, SubT, SubD} <: AbstractHessian
    t1sseeds::VP
    t1sF::VD
    x::VT
    t1sx::VD
    map::VI
    # Cache views on x and its dual vector to avoid reallocating on the GPU
    varx::SubT
    t1svarx::SubD
    function Hessian(structure, F, vm, va, ybus_re, ybus_im, pinj, qinj, pv, pq, ref)
        nvbus = length(vm)
        npbus = length(pinj)
        nref = length(ref)
        if F isa Array
            VI = Vector{Int}
            VT = Vector{Float64}
            MT = Matrix{Float64}
            SMT = SparseMatrixCSC
            A = Vector
        elseif F isa CUDA.CuArray
            VI = CUDA.CuVector{Int}
            VT = CUDA.CuVector{Float64}
            MT = CUDA.CuMatrix{Float64}
            SMT = CUSPARSE.CuSparseMatrixCSR
            A = CUDA.CuVector
        else
            error("Wrong array type ", typeof(F))
        end

        map = VI(structure.map)
        nmap = length(structure.map)
        t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
        x = VT(zeros(Float64, 2*nvbus + npbus))
        t1sx = A{t1s{1}}(x)
        t1sF = A{t1s{1}}(zeros(Float64, length(F)))
        t1sseeds = A{ForwardDiff.Partials{1,Float64}}(undef, nmap)
        varx = view(x, map)
        t1svarx = view(t1sx, map)
        VP = typeof(t1sseeds)
        VD = typeof(t1sx)
        return new{VI, VT, MT, SMT, VP, VD, typeof(varx), typeof(t1svarx)}(
            t1sseeds, t1sF, x, t1sx, map, varx, t1svarx
        )
    end
end

"""
    seed_kernel!

Seeding on GPU parallelized over the `ncolor` number of duals

"""
@kernel function seed_kernel!(
    duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
    seeds::AbstractArray{ForwardDiff.Partials{N,V}}
) where {T,V,N}
    i = @index(Global, Linear)
    duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
end

"""
    seed_kernel!

Calling the GPU seeding kernel

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

"""
    getpartials_kernel_cpu!(compressedJ, t1sF)

Extract the partials from the AutoDiff dual type on the CPU and put it in the
compressed Jacobian

"""
@kernel function getpartials_kernel_cpu!(compressedJ, t1sF)
    i = @index(Global, Linear)
    compressedJ[:, i] .= ForwardDiff.partials.(t1sF[i]).values
end

"""
    getpartials_kernel_gpu!(compressedJ, t1sF)

Extract the partials from the AutoDiff dual type on the GPU and put it in the
compressed Jacobian

"""
@kernel function getpartials_kernel_gpu!(compressedJ, t1sF)
    i = @index(Global, Linear)
    for j in eachindex(ForwardDiff.partials.(t1sF[i]).values)
        @inbounds compressedJ[j, i] = ForwardDiff.partials.(t1sF[i]).values[j]
    end
end

"""
    getpartials_kernel!(compressedJ, t1sF, nbus)

Calling the partial extraction kernel

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

"""
    uncompress_kernel_gpu!(J_rowPtr, J_colVal, J_nzVal, compressedJ, coloring, nmap)

Uncompress the compressed Jacobian matrix from `compressedJ` to sparse CSR on
the GPU. 
"""
@kernel function uncompress_kernel_gpu!(J_rowPtr, J_colVal, J_nzVal, compressedJ, coloring)
    i = @index(Global, Linear)
    @inbounds for j in J_rowPtr[i]:J_rowPtr[i+1]-1
        @inbounds J_nzVal[j] = compressedJ[coloring[J_colVal[j]], i]
    end
end

"""
    uncompress_kernel_gpu!(J_colptr, J_rowval, J_nzval, compressedJ, coloring, nmap)

Uncompress the compressed Jacobian matrix from `compressedJ` to sparse CSC on
the CPU.
"""
@kernel function uncompress_kernel_cpu!(J_colptr, J_rowval, J_nzval, compressedJ, coloring)
    # CSC is column oriented: nmap is equal to number of columns
    i = @index(Global, Linear)
    @inbounds for j in J_colptr[i]:J_colptr[i+1]-1
        @inbounds J_nzval[j] = compressedJ[coloring[i], J_rowval[j]]
    end
end

"""
    uncompress_kernel!(J, compressedJ, coloring)

Uncompress the compressed Jacobian matrix from `compressedJ` to sparse CSC or CSR.
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
    residual_jacobian!(J::Jacobian,
                        residual_polar!,
                        vm, va, ybus_re, ybus_im, pinj, qinj, pv, pq, ref,
                        nbus, type::AbstractJacobian)

Update the sparse Jacobian entries using AutoDiff. No allocations are taking place in this function.

* `J::Jacobian`: Factory created Jacobian object to update
* `residual_polar`: Primal function
* `vm, va, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus`: Inputs both
  active and passive parameters. Active inputs are mapped to `x` via the preallocated views.
* `type::AbstractJacobian`: Either `StateJacobian` or `ControlJacobian`
"""
function residual_jacobian!(J::Jacobian,
                             residual_polar!,
                             vm, va, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus,
                             type::AbstractJacobian)
    nvbus = length(vm)
    ninj = length(pinj)
    if isa(type, StateJacobian)
        J.x[1:nvbus] .= vm
        J.x[nvbus+1:2*nvbus] .= va
        J.t1sx .= J.x
        J.t1sF .= 0.0
    elseif isa(type, ControlJacobian)
        J.x[1:nvbus] .= vm
        J.x[nvbus+1:nvbus+ninj] .= pinj
        J.t1sx .= J.x
        J.t1sF .= 0.0
    else
        error("Unsupported Jacobian structure")
    end

    seed!(J.t1sseeds, J.varx, J.t1svarx, nbus)

    if isa(type, StateJacobian)
        residual_polar!(
            J.t1sF,
            view(J.t1sx, 1:nvbus),
            view(J.t1sx, nvbus+1:2*nvbus),
            ybus_re, ybus_im,
            pinj, qinj,
            pv, pq, nbus
        )
    elseif isa(type, ControlJacobian)
        residual_polar!(
            J.t1sF,
            view(J.t1sx, 1:nvbus),
            va,
            ybus_re, ybus_im,
            view(J.t1sx, nvbus+1:nvbus+ninj), qinj,
            pv, pq, nbus
        )
    else
        error("Unsupported Jacobian structure")
    end

    getpartials_kernel!(J.compressedJ, J.t1sF, nbus)
    uncompress_kernel!(J.J, J.compressedJ, J.coloring)

    return nothing
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
function residual_hessian_adj_tgt!(H::Hessian,
                             residual_adj_polar!,
                             lambda, tgt,
                             v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus)
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
    residual_adj_polar!(
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

function Base.show(io::IO, jacobian::AbstractJacobian)
    ncolor = size(unique(jacobian.coloring), 1)
    print(io, "Number of Jacobian colors: ", ncolor)
end

end
