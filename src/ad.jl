
module AD

using CUDA
using CUDA.CUSPARSE
using ForwardDiff
using KernelAbstractions
using SparseArrays
using TimerOutputs
using SparsityDetection
using SparseDiffTools
using ..ExaPF: Spmat, xzeros

import Base: show

"""
    AbstractADFramework

Automatic differentiation for the compressed Jacobians of the
constraints `g(x,u)` with respect to the state `x` and the control `u`
(here called design).

TODO: Use dispatch to unify the code of the state and control Jacobian.
This is currently not done because the abstraction of the indexing is not yet resolved.

"""
abstract type AbstractADFramework end
abstract type AbstractObjectiveAD <: AbstractADFramework end
abstract type AbstractJacobianAD end
abstract type AbstractHessianAD end

t1s{N,V} = ForwardDiff.Dual{Nothing,V, N} where {N,V}
t2s{M,N,V} =  ForwardDiff.Dual{Nothing,t1s{N,V}, M} where {M,N,V}

function _init_seed!(t1sseeds::Array{ForwardDiff.Partials{N,Float64},1}, coloring, ncolor, nmap) where {N}
    t1sseedvec = zeros(Float64, ncolor)
    @inbounds for i in 1:nmap
        for j in 1:ncolor
            if coloring[i] == j
                t1sseedvec[j] = 1.0
            end
        end
        t1sseeds[i] = ForwardDiff.Partials{ncolor, Float64}(NTuple{ncolor, Float64}(t1sseedvec))
        t1sseedvec .= 0
    end
end

function _init_seed!(t2sseeds::Array{ForwardDiff.Partials{M,t1s{N,V}},1}, coloring, ncolor, nmap) where {M,N,V}
    t2sseedvec = zeros(t1s{N,V}, nmap)
    @inbounds for i in 1:nmap
        for j in 1:nmap
            t2sseedvec[j] = 1.0
        end
        t2sseeds[i] = ForwardDiff.Partials{nmap, t1s{N,V}}(NTuple{nmap, t1s{N,V}}(t2sseedvec))
        t2sseedvec .= 0
    end
end

"""
    StateJacobianAD

Creates an object for the state Jacobian

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
struct StateJacobianAD{VI, VT, MT, SMT, VP, VD, SubT, SubD} <: AbstractJacobianAD
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
    function StateJacobianAD(F, v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus)
        nv_m = size(v_m, 1)
        nv_a = size(v_a, 1)
        if F isa Array
            VI = Vector{Int}
            VT = Vector{Float64}
            MT = Matrix{Float64}
            SMT = SparseMatrixCSC
            A = Vector
        elseif F isa CuArray
            VI = CuVector{Int}
            VT = CuVector{Float64}
            MT = CuMatrix{Float64}
            SMT = CuSparseMatrixCSR
            A = CuVector
        else
            error("Wrong array type ", typeof(F))
        end

        mappv = [i + nv_m for i in pv]
        mappq = [i + nv_m for i in pq]
        # Ordering for x is (θ_pv, θ_pq, v_pq)
        map = VI(vcat(mappv, mappq, pq))
        nmap = size(map,1)

        # Used for sparsity detection with randomized inputs
        function residualJacobian(V, Ybus, pv, pq)
            n = size(V, 1)
            Ibus = Ybus*V
            diagV       = sparse(1:n, 1:n, V, n, n)
            diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
            diagVnorm   = sparse(1:n, 1:n, V./abs.(V), n, n)

            dSbus_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
            dSbus_dVa = 1im * diagV * conj(diagIbus - Ybus * diagV)

            j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
            j12 = real(dSbus_dVm[[pv; pq], pq])
            j21 = imag(dSbus_dVa[pq, [pv; pq]])
            j22 = imag(dSbus_dVm[pq, pq])

            J = [j11 j12; j21 j22]
        end

        # Need a host arrays for the sparsity detection below
        spmap = Vector(map)
        hybus_re = Spmat{Vector{Int}, Vector{Float64}}(ybus_re)
        hybus_im = Spmat{Vector{Int}, Vector{Float64}}(ybus_im)
        n = nv_a
        Yre = SparseMatrixCSC{Float64,Int64}(n, n, hybus_re.colptr, hybus_re.rowval, hybus_re.nzval)
        Yim = SparseMatrixCSC{Float64,Int64}(n, n, hybus_im.colptr, hybus_im.rowval, hybus_im.nzval)
        Y = Yre .+ 1im .* Yim
        # Randomized inputs
        Vre = Float64.([i for i in 1:n])
        Vim = Float64.([i for i in n+1:2*n])
        V = Vre .+ 1im .* Vim
        J = residualJacobian(V, Y, pv, pq)
        coloring = VI(matrix_colors(J))
        ncolor = size(unique(coloring),1)
        if F isa CuArray
            J = CuSparseMatrixCSR(J)
        end
        x = VT(zeros(Float64, nv_m + nv_a))
        t1sx = A{t1s{ncolor,Float64}}(x)
        t1sF = A{t1s{ncolor,Float64}}(zeros(Float64, nmap))

        t1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(undef, nmap)
        _init_seed!(t1sseeds, coloring, ncolor, nmap)

        compressedJ = MT(zeros(Float64, ncolor, nmap))
        nthreads=256
        nblocks=ceil(Int64, nmap/nthreads)
        # Views
        varx = view(x, map)
        t1svarx = view(t1sx, map)
        VP = typeof(t1sseeds)
        VD = typeof(t1sx)
        return new{VI, VT, MT, SMT, VP, VD, typeof(varx), typeof(t1svarx)}(
            J, compressedJ, coloring, t1sseeds, t1sF, x, t1sx, map, varx, t1svarx
        )
    end
end

"""
    DesignJacobianAD

Creates an object for the control Jacobian.

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
struct DesignJacobianAD{VI, VT, MT, SMT, VP, VD, SubT, SubD} <: AbstractJacobianAD
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
    function DesignJacobianAD(F, v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus)
        nv_m = size(v_m, 1)
        nv_a = size(v_a, 1)
        npbus = size(pinj, 1)
        nref = size(ref, 1)
        # ncolor = size(unique(coloring),1)
        if F isa Array
            VI = Vector{Int}
            VT = Vector{Float64}
            MT = Matrix{Float64}
            SMT = SparseMatrixCSC
            A = Vector
        elseif F isa CuArray
            VI = CuVector{Int}
            VT = CuVector{Float64}
            MT = CuMatrix{Float64}
            SMT = CuSparseMatrixCSR
            A = CuVector
        else
            error("Wrong array type ", typeof(F))
        end

        mappv =  [i + nv_a for i in pv]
        map = VI(vcat(ref, mappv, pv))
        nmap = size(map,1)

        # Used for sparsity detection with randomized inputs
        function residualJacobian(V, Ybus, pinj, qinj, ref, pv, pq)
            n = size(V, 1)
            Ibus = Ybus*V
            diagV       = sparse(1:n, 1:n, V, n, n)
            diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
            diagVnorm   = sparse(1:n, 1:n, V./abs.(V), n, n)

            dSbus_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
            dSbus_dpbus = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm

            j11 = real(dSbus_dVm[[pv; pq], [ref; pv; pv]])
            j21 = imag(dSbus_dVm[pq, [ref; pv; pv]])
            J = [j11; j21]
        end

        # Need a host arrays for the sparsity detection below
        spmap = Vector(map)
        hybus_re = Spmat{Vector{Int}, Vector{Float64}}(ybus_re)
        hybus_im = Spmat{Vector{Int}, Vector{Float64}}(ybus_im)
        n = nv_a
        Yre = SparseMatrixCSC{Float64,Int64}(n, n, hybus_re.colptr, hybus_re.rowval, hybus_re.nzval)
        Yim = SparseMatrixCSC{Float64,Int64}(n, n, hybus_im.colptr, hybus_im.rowval, hybus_im.nzval)
        Y = Yre .+ 1im .* Yim
        # Randomized inputs
        Vre = Float64.([i for i in 1:n])
        Vim = Float64.([i for i in n+1:2*n])
        V = Vre .+ 1im .* Vim
        J = residualJacobian(V, Y, pinj, qinj, ref, pv, pq)
        coloring = VI(matrix_colors(J))
        ncolor = size(unique(coloring),1)
        if F isa CuArray
            J = CuSparseMatrixCSR(J)
        end
        x = xzeros(VT, npbus + nv_a)
        t1sx = A{t1s{ncolor,Float64}}(x)
        # t1sF = T{t1s{ncolor}}(undef, nmap)
        t1sF = A{t1s{ncolor,Float64}}(zeros(Float64, length(F)))
        t1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(undef, nmap)
        _init_seed!(t1sseeds, coloring, ncolor, nmap)

        compressedJ = MT(zeros(Float64, ncolor, length(F)))
        nthreads=256
        nblocks=ceil(Int64, nmap/nthreads)
        # Views
        varx = view(x, map)
        t1svarx = view(t1sx, map)
        VP = typeof(t1sseeds)
        VD = typeof(t1sx)
        return new{VI, VT, MT, SMT, VP, VD, typeof(varx), typeof(t1svarx)}(
            J, compressedJ, coloring, t1sseeds, t1sF, x, t1sx, map, varx, t1svarx
        )
    end
end

"""
    myseed_kernel_cpu

Seeding on the CPU, not parallelized.

"""
function myseed_kernel_cpu(
    duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
    seeds::AbstractArray{ForwardDiff.Partials{N,V}}
) where {T,V,N}
    for i in 1:size(duals,1)
        duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
    end
end

"""
    myseed_kernel_gpu

Seeding on GPU parallelized over the `ncolor` number of duals

"""
function myseed_kernel_gpu(
    duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
    seeds::AbstractArray{ForwardDiff.Partials{N,V}}
) where {T,V,N}
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:size(duals,1)
        duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
    end
end

"""
    seeding(t1sseeds::CuVector{ForwardDiff.Partials{N,V}}, varx, t1svarx, nbus) where {N, V}

Calling the GPU seeding kernel

"""
function seeding(t1sseeds::CuVector{ForwardDiff.Partials{N,V}}, varx, t1svarx, nbus) where {N, V}
    nthreads = 256
    nblocks = div(nbus, nthreads, RoundUp)
    CUDA.@sync begin
        @cuda threads=nthreads blocks=nblocks myseed_kernel_gpu(
            t1svarx,
            varx,
            t1sseeds,
        )
    end
end

"""
    seeding(t1sseeds::Vector{ForwardDiff.Partials{N,V}}, varx, t1svarx, nbus) where {N, V}

Calling the CPU seeding kernel

"""
function seeding(t1sseeds::Vector{ForwardDiff.Partials{N,V}}, varx, t1svarx, nbus) where {N, V}
    myseed_kernel_cpu(t1svarx, varx, t1sseeds)
end

"""
    getpartials_cpu(compressedJ, t1sF)

Extract the partials from the AD dual type on the CPU and put it in the
compressed Jacobian

"""
function getpartials_cpu(compressedJ, t1sF)
    for i in 1:size(t1sF,1) # Go over outputs
        compressedJ[:, i] .= ForwardDiff.partials.(t1sF[i]).values
    end
end

"""
    getpartials_gpu(compressedJ, t1sF)

Extract the partials from the AD dual type on the GPU and put it in the
compressed Jacobian

"""
function getpartials_gpu(compressedJ, t1sF)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:size(t1sF, 1) # Go over outputs
        for j in eachindex(ForwardDiff.partials.(t1sF[i]).values)
            @inbounds compressedJ[j, i] = ForwardDiff.partials.(t1sF[i]).values[j]
        end
    end
end

"""
    getpartials(compressedJ::CuArray{T, 2}, t1sF, nbus) where T

Calling the GPU partial extraction kernel

"""
function getpartials(compressedJ::CuArray{T, 2}, t1sF, nbus) where T
    nthreads = 256
    nblocks = div(nbus, nthreads, RoundUp)
    CUDA.@sync begin
        @cuda threads=nthreads blocks=nblocks getpartials_gpu(
            compressedJ,
            t1sF
        )
    end
end

"""
    getpartials(compressedJ::Array{T, 2}, t1sF, nbus) where T

Calling the GPU partial extraction kernel

"""
function getpartials(compressedJ::Array{T, 2}, t1sF, nbus) where T
    getpartials_cpu(compressedJ, t1sF)
end

"""
    _uncompress(J_nzVal, J_rowPtr, J_colVal, compressedJ, coloring, nmap)

Uncompress the compressed Jacobian matrix from `compressedJ` to sparse CSR on
the GPU. Only bitarguments are allowed for the kernel.
(for GPU only) TODO: should convert to @kernel
"""
function _uncompress(J_nzVal, J_rowPtr, J_colVal, compressedJ, coloring, nmap)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:nmap
        for j in J_rowPtr[i]:J_rowPtr[i+1]-1
            @inbounds J_nzVal[j] = compressedJ[coloring[J_colVal[j]], i]
        end
    end
end

"""
    uncompress!(J::SparseArrays.SparseMatrixCSC, compressedJ, coloring)

Uncompress the compressed Jacobian matrix from `compressedJ` to sparse CSC on
the CPU.
"""
function uncompress!(J::SparseArrays.SparseMatrixCSC, compressedJ, coloring)
    # CSC is column oriented: nmap is equal to number of columns
    nmap = size(J, 2)
    # TODO: coloring[i] leads to an out of bounds access here. Added the @assert here to observe.
    if maximum(coloring) != size(compressedJ,1)
        @show coloring
        @show J.colptr
        @show J.rowval
        @show J.nzval
        @show length(J.nzval)
        @show size(J)
        @assert(maximum(coloring) == size(compressedJ,1))
    end
    for i in 1:nmap
        for j in J.colptr[i]:J.colptr[i+1]-1
            @inbounds J.nzval[j] = compressedJ[coloring[i], J.rowval[j]]
        end
    end
end

"""
    uncompress!(J::CUDA.CUSPARSE.CuSparseMatrixCSR, compressedJ, coloring)

Uncompress the compressed Jacobian matrix from `compressedJ` to sparse CSC on
the GPU by calling the kernel [`_uncompress`](@ref).
"""
function uncompress!(J::CUDA.CUSPARSE.CuSparseMatrixCSR, compressedJ, coloring)
    # CSR is row oriented: nmap is equal to number of rows
    nmap = size(J, 1)
    nthreads = 256
    nblocks = div(nmap, nthreads, RoundUp)
    CUDA.@sync begin
        @cuda threads=nthreads blocks=nblocks _uncompress(
                J.nzVal,
                J.rowPtr,
                J.colVal,
                compressedJ,
                coloring, nmap
        )
    end
end

"""
    residualJacobianAD!(arrays::StateJacobianAD,
                        residualFunction_polar!,
                        v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus,
                        timer = nothing)

Update the sparse Jacobian entries using AD. No allocations are taking place in this function.

* `arrays::StateJacobianAD`: Factory created Jacobian object to update
* `residualFunction_polar`: Primal function
* `v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus`: Inputs both
  active and passive parameters. Active inputs are mapped to `x` via the preallocated views.

"""
function residualJacobianAD!(arrays::StateJacobianAD,
                             residualFunction_polar!,
                             v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus,
                             timer = nothing)
    @timeit timer "Before" begin
        @timeit timer "Setup" begin
            nv_m = size(v_m, 1)
            nv_a = size(v_a, 1)
            nmap = size(arrays.map, 1)
            n = nv_m + nv_a
        end
        @timeit timer "Arrays" begin
            arrays.x[1:nv_m] .= v_m
            arrays.x[nv_m+1:nv_m+nv_a] .= v_a
            arrays.t1sx .= arrays.x
            arrays.t1sF .= 0.0
        end
    end
    @timeit timer "Seeding" begin
        seeding(arrays.t1sseeds, arrays.varx, arrays.t1svarx, nbus)
    end

    @timeit timer "Function" begin
        residualFunction_polar!(
            arrays.t1sF,
            view(arrays.t1sx, 1:nv_m),
            view(arrays.t1sx, nv_m+1:nv_m+nv_a),
            ybus_re, ybus_im,
            pinj, qinj,
            pv, pq, nbus
        )
    end

    @timeit timer "Get partials" begin
        getpartials(arrays.compressedJ, arrays.t1sF, nbus)
    end
    @timeit timer "Uncompress" begin
        uncompress!(arrays.J, arrays.compressedJ, arrays.coloring)
    end
    return nothing
end

"""
    residualJacobianAD!(arrays::DesignJacobianAD,
                        residualFunction_polar!,
                        v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus,
                        timer = nothing)

Update the sparse Jacobian entries using AD. No allocations are taking place in this function.

* `arrays::DesignJacobianAD`: Factory created Jacobian object to update
* `residualFunction_polar`: Primal function
* `v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus`: Inputs both active and passive parameters. Active inputs are mapped to `x` via the preallocated views.

"""
function residualJacobianAD!(arrays::DesignJacobianAD,
                             residualFunction_polar!,
                             v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus,
                             timer = nothing)

    @timeit timer "Before" begin
        @timeit timer "Setup" begin
            npinj = size(pinj , 1)
            nv_m = size(v_m, 1)
            nmap = size(arrays.map, 1)
            n = npinj + nv_m
        end
        @timeit timer "Arrays" begin
            arrays.x[1:nv_m] .= v_m
            arrays.x[nv_m+1:nv_m+npinj] .= pinj
            arrays.t1sx .= arrays.x
            arrays.t1sF .= 0.0
        end
    end
    @timeit timer "Seeding" begin
        seeding(arrays.t1sseeds, arrays.varx, arrays.t1svarx, nbus)
    end
    @timeit timer "Function" begin
        residualFunction_polar!(
            arrays.t1sF,
            view(arrays.t1sx, 1:nv_m),
            v_a,
            ybus_re, ybus_im,
            view(arrays.t1sx, nv_m+1:nv_m + npinj), qinj,
            pv, pq, nbus
        )
    end

    @timeit timer "Get partials" begin
        getpartials(arrays.compressedJ, arrays.t1sF, nbus)
    end
    @timeit timer "Uncompress" begin
        # Uncompress matrix. Sparse matrix elements have different names with CUDA
        uncompress!(arrays.J, arrays.compressedJ, arrays.coloring)
    end
end

function Base.show(io::IO, ad::AbstractJacobianAD)
    ncolor = size(unique(ad.coloring), 1)
    print(io, "Number of Jacobian colors: ", ncolor)
end

struct ObjectiveAD{VT}
    ∇fₓ::VT
    ∇fᵤ::VT
    ∂pg::VT
    ∂vm::VT
    ∂va::VT
    jvₓ::VT
    jvᵤ::VT
end


"""
    StateHessianAD

Creates an object for the state Jacobian

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
struct StateHessianAD{VI, VT, MT, SMT, VP, VP2, VD, SubT, SubD} <: AbstractHessianAD
    H::Array{SMT}
    compressedH::Array{MT}
    coloring::VI
    t1sseeds::VP
    t2sseeds::VP2
    t2sF::VD
    x::VT
    t2sx::VD
    map::VI
    # Cache views on x and its dual vector to avoid reallocating on the GPU
    varx::SubT
    t2svarx::SubD
    function StateHessianAD(F, v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus)
        nv_m = size(v_m, 1)
        nv_a = size(v_a, 1)
        if F isa Array
            VI = Vector{Int}
            VT = Vector{Float64}
            MT = Matrix{Float64}
            SMT = SparseMatrixCSC
            A = Vector
        elseif F isa CuArray
            VI = CuVector{Int}
            VT = CuVector{Float64}
            MT = CuMatrix{Float64}
            SMT = CuSparseMatrixCSR
            A = CuVector
        else
            error("Wrong array type ", typeof(F))
        end

        mappv = [i + nv_m for i in pv]
        mappq = [i + nv_m for i in pq]
        # Ordering for x is (θ_pv, θ_pq, v_pq)
        map = VI(vcat(mappv, mappq, pq))
        nmap = size(map,1)

        # Used for sparsity detection with randomized inputs
        function residualJacobian(V, Ybus, pv, pq)
            n = size(V, 1)
            Ibus = Ybus*V
            diagV       = sparse(1:n, 1:n, V, n, n)
            diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
            diagVnorm   = sparse(1:n, 1:n, V./abs.(V), n, n)

            dSbus_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
            dSbus_dVa = 1im * diagV * conj(diagIbus - Ybus * diagV)

            j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
            j12 = real(dSbus_dVm[[pv; pq], pq])
            j21 = imag(dSbus_dVa[pq, [pv; pq]])
            j22 = imag(dSbus_dVm[pq, pq])

            J = [j11 j12; j21 j22]
        end

        # Need a host arrays for the sparsity detection below
        spmap = Vector(map)
        hybus_re = Spmat{Vector{Int}, Vector{Float64}}(ybus_re)
        hybus_im = Spmat{Vector{Int}, Vector{Float64}}(ybus_im)
        n = nv_a
        Yre = SparseMatrixCSC{Float64,Int64}(n, n, hybus_re.colptr, hybus_re.rowval, hybus_re.nzval)
        Yim = SparseMatrixCSC{Float64,Int64}(n, n, hybus_im.colptr, hybus_im.rowval, hybus_im.nzval)
        Y = Yre .+ 1im .* Yim
        # Randomized inputs
        Vre = Float64.([i for i in 1:n])
        Vim = Float64.([i for i in n+1:2*n])
        V = Vre .+ 1im .* Vim
        J = residualJacobian(V, Y, pv, pq)
        coloring = VI(matrix_colors(J))
        ncolor = size(unique(coloring),1)
        if F isa CuArray
            J = CuSparseMatrixCSR(J)
        end
        H = Array{SMT}(undef, nmap)
        for i in 1:nmap
            H[i] = copy(J)
        end
        x = VT(zeros(Float64, nv_m + nv_a))
        t2sx = A{t2s{nmap,ncolor,Float64}}(x)
        t2sF = A{t2s{nmap,ncolor,Float64}}(zeros(Float64, nmap))
        t1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(undef, nmap)
        @show typeof(t1sseeds)
        t2sseeds = A{ForwardDiff.Partials{nmap,t1s{ncolor,Float64}}}(undef, nmap)
        @show typeof(t2sseeds)
        _init_seed!(t2sseeds, coloring, ncolor, nmap)

        compressedH = Array{MT}(undef, nmap)
        for i in 1:nmap
            compressedH[i] = copy(MT(zeros(Float64, ncolor, nmap)))
        end
        nthreads=256
        nblocks=ceil(Int64, nmap/nthreads)
        # Views
        varx = view(x, map)
        t2svarx = view(t2sx, map)
        VP = typeof(t1sseeds)
        VP2 = typeof(t2sseeds)
        VD = typeof(t2sx)
        return new{VI, VT, MT, SMT, VP, VP2, VD, typeof(varx), typeof(t2svarx)}(
            H, compressedH, coloring, t1sseeds, t2sseeds, t2sF, x, t2sx, map, varx, t2svarx
        )
    end
end

function seeding(t2sseeds, t1sseeds::Vector{ForwardDiff.Partials{N,V}}, varx, t2svarx, nbus) where {N, V}
    @show typeof(t2sseeds)
    @show typeof(t1sseeds)
    myseed_kernel_cpu(t2svarx, varx, t2sseeds, t1sseeds)
end

function myseed_kernel_cpu(
    duals::AbstractArray{ForwardDiff.Dual{T,t1s{N,V},M}}, 
    x,
    t2sseeds::Array{ForwardDiff.Partials{M,t1s{N,V}},1},
    t1sseeds::Vector{ForwardDiff.Partials{N,V}}
) where {T,V,M,N}
    @show M, N
    for i in 1:size(duals,1)
        # duals[i] = ForwardDiff.Dual{T,t1s{N},N}(ForwardDiff.Dual{T,V,N}(x[i], t1sseeds[i]), t2sseeds[i])
        @show typeof(duals[i])
        @show typeof(ForwardDiff.Dual{T,V,N}(x[i]))
        # duals[i] = ForwardDiff.Dual{T,t1s{N,V},M}(ForwardDiff.Dual{T,V,N}(x[i], t1sseeds[i]))
        duals[i] = ForwardDiff.Dual{T,t1s{N,V},M}(ForwardDiff.Dual{T,V,N}(x[i]))
    end
end

function residualHessianAD!(arrays::StateHessianAD,
                             residualFunction_polar!,
                             v_m, v_a, ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus,
                             timer = nothing)
    @timeit timer "Before" begin
        @timeit timer "Setup" begin
            nv_m = size(v_m, 1)
            nv_a = size(v_a, 1)
            nmap = size(arrays.map, 1)
            n = nv_m + nv_a
        end
        @timeit timer "Arrays" begin
            arrays.x[1:nv_m] .= v_m
            arrays.x[nv_m+1:nv_m+nv_a] .= v_a
            arrays.t2sx .= arrays.x
            arrays.t2sF .= 0.0
        end
    end
    @timeit timer "Seeding" begin
        @show typeof(arrays.t1sseeds)
        seeding(arrays.t2sseeds, arrays.t1sseeds, arrays.varx, arrays.t2svarx, nbus)
    end

    @timeit timer "Function" begin
        residualFunction_polar!(
            arrays.t2sF,
            view(arrays.t2sx, 1:nv_m),
            view(arrays.t2sx, nv_m+1:nv_m+nv_a),
            ybus_re, ybus_im,
            pinj, qinj,
            pv, pq, nbus
        )
    end

    @timeit timer "Get partials" begin
        getpartials(arrays.compressedH, arrays.t2sF, nbus)
    end
    @timeit timer "Uncompress" begin
        uncompress!(arrays.H, arrays.compressedH, arrays.coloring)
    end
    return nothing
end

function getpartials(compressedH::Array{Array{T, 2}}, t2sF, nbus) where T
    @show size(compressedH)
    @show size(compressedH[:,1])
    @show size(compressedH[1,:])
    @show size(t2sF)
    for idx in 1:length(compressedH)
        getpartials_cpu(compressedH, t2sF, idx)
    end
end

function getpartials_cpu(compressedH, t2sF, idx)
    # @show typeof(t2sF)
    # @show typeof(ForwardDiff.partials.((ForwardDiff.partials.(t2sF[1])).values[1]).values)

    @show typeof(ForwardDiff.partials.(t2sF[1]))
    # @show length(ForwardDiff.partials.(t2sF[1]))
    @show ForwardDiff.partials.((ForwardDiff.partials.(t2sF[idx])).values[12]).values
    for i in 1:size(t2sF,1) # Go over outputs
        # @show typeof(ForwardDiff.partials.(t2sF[i]))
        # @show typeof(ForwardDiff.partials.(t2sF[i]).values)
        # compressedH[idx][:, i] .= ForwardDiff.partials.(t2sF[i]).values
        # compressedH[idx][:, i] .= ForwardDiff.partials.(t2sF[i]).values
        # @show typeof(ForwardDiff.partials.((ForwardDiff.partials.(t2sF[idx])).values[i]).values)
        # @show typeof(ForwardDiff.partials.((ForwardDiff.partials.(t2sF[idx])).values[i]))
        # @show length(ForwardDiff.partials.((ForwardDiff.partials.(t2sF[idx])).values[i]).values)
        # @show typeof(compressedH[idx][:,i])
        # @show length(compressedH[idx][:,i])
        compressedH[idx][:,i] .= ForwardDiff.partials.((ForwardDiff.partials.(t2sF[idx])).values[i]).values
    end
end

end