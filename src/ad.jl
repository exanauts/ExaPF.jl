module AD

using CUDA
using CUDA.CUSPARSE
using ForwardDiff
using KernelAbstractions
using SparseArrays
using TimerOutputs
using SparsityDetection
using SparseDiffTools
using ..ExaPF: Spmat

import Base: show

abstract type AbstractJacobianAD end

struct StateJacobianAD{VI, VT, MT, SMT, VP, VD} <: AbstractJacobianAD
    J::SMT
    compressedJ::MT
    coloring::VI
    t1sseeds::VP
    t1sF::VD
    x::VT
    t1sx::VD
    map::VI
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
        map = VI(vcat(pq, mappq, mappv))
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

            j11 = real(dSbus_dVm[[pv; pq], pq])
            j12 = real(dSbus_dVa[[pv; pq], [pq; pv]])
            j21 = imag(dSbus_dVm[pq, pq])
            j22 = imag(dSbus_dVa[pq, [pq; pv]])

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
        Vre = rand(n)
        Vim = rand(n)
        V = Vre .+ 1im .* Vim
        J = residualJacobian(V, Y, pv, pq)
        coloring = VI(matrix_colors(J))
        ncolor = size(unique(coloring),1)
        if F isa CuArray
            J = CuSparseMatrixCSR(J)
        end
        t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
        # x = T{Float64}(undef, nv_m + nv_a)
        x = VT(zeros(Float64, nv_m + nv_a))
        t1sx = A{t1s{ncolor}}(x)
        # t1sF = T{t1s{ncolor}}(undef, nmap)
        t1sF = A{t1s{ncolor}}(zeros(Float64, nmap))
        t1sseedvec = zeros(Float64, ncolor)
        t1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(undef, nmap)
        for i in 1:nmap
            for j in 1:ncolor
                if coloring[i] == j
                    t1sseedvec[j] = 1.0
                end
            end
            t1sseeds[i] = ForwardDiff.Partials{ncolor, Float64}(NTuple{ncolor, Float64}(t1sseedvec))
            t1sseedvec .= 0
        end
        compressedJ = MT(zeros(Float64, ncolor, nmap))
        nthreads=256
        nblocks=ceil(Int64, nmap/nthreads)
        VP = typeof(t1sseeds)
        VD = typeof(t1sx)
        return new{VI, VT, MT, SMT, VP, VD}(J, compressedJ, coloring, t1sseeds, t1sF, x, t1sx, map)
    end
end

struct DesignJacobianAD{VI, VT, MT, SMT, VP, VD} <: AbstractJacobianAD
    J::SMT
    compressedJ::MT
    coloring::VI
    t1sseeds::VP
    t1sF::VD
    x::VT
    t1sx::VD
    map::VI
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
        Vre = rand(n)
        Vim = rand(n)
        V = Vre .+ 1im .* Vim
        J = residualJacobian(V, Y, pinj, qinj, ref, pv, pq)
        coloring = VI(matrix_colors(J))
        ncolor = size(unique(coloring),1)
        if F isa CuArray
            J = CuSparseMatrixCSR(J)
        end
        t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
        # x = T{Float64}(undef, nv_m + nv_a)
        x = VT(zeros(Float64, npbus + nv_a))
        ncolor = length(x)
        t1sx = A{t1s{ncolor}}(x)
        # t1sF = T{t1s{ncolor}}(undef, nmap)
        t1sF = A{t1s{ncolor}}(zeros(Float64, length(F)))
        t1sseedvec = zeros(Float64, ncolor)
        t1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(undef, nmap)
        for i in 1:nmap
            for j in 1:ncolor
                if coloring[i] == j
                    t1sseedvec[j] = 1.0
                end
            end
            t1sseeds[i] = ForwardDiff.Partials{ncolor, Float64}(NTuple{ncolor, Float64}(t1sseedvec))
            t1sseedvec .= 0
        end
        compressedJ = MT(zeros(Float64, ncolor, length(F)))
        nthreads=256
        nblocks=ceil(Int64, nmap/nthreads)
        VP = typeof(t1sseeds)
        VD = typeof(t1sx)
        return new{VI, VT, MT, SMT, VP, VD}(J, compressedJ, coloring, t1sseeds, t1sF, x, t1sx, map)
    end
end

function myseed_kernel_cpu(
    duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
    seeds::AbstractArray{ForwardDiff.Partials{N,V}}
) where {T,V,N}
    for i in 1:size(duals,1)
        duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
    end
end

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

function getpartials_cpu(compressedJ, t1sF)
    for i in 1:size(t1sF,1) # Go over outputs
        compressedJ[:, i] .= ForwardDiff.partials.(t1sF[i]).values
    end
end

function getpartials_gpu(compressedJ, t1sF)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:size(t1sF, 1) # Go over outputs
        for j in eachindex(ForwardDiff.partials.(t1sF[i]).values)
            @inbounds compressedJ[j, i] = ForwardDiff.partials.(t1sF[i]).values[j]
        end
    end
end

# uncompress (for GPU only)
# TODO: should convert to @kernel
function uncompress(J_nzVal, J_rowPtr, J_colVal, compressedJ, coloring, nmap)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:nmap
        for j in J_rowPtr[i]:J_rowPtr[i+1]-1
            @inbounds J_nzVal[j] = compressedJ[coloring[J_colVal[j]], i]
        end
    end
end

function residualJacobianAD!(arrays::StateJacobianAD, residualFunction_polar!, v_m, v_a,
                             ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus, timer = nothing)
    device = isa(arrays.J, SparseArrays.SparseMatrixCSC) ? CPU() : CUDADevice()
    @timeit timer "Before" begin
        @timeit timer "Setup" begin
            nv_m = size(v_m, 1)
            nv_a = size(v_a, 1)
            nmap = size(arrays.map, 1)
            nthreads=256
            nblocks=ceil(Int64, nmap/nthreads)
            n = nv_m + nv_a
        end
        @timeit timer "Arrays" begin
            arrays.x[1:nv_m] .= v_m
            arrays.x[nv_m+1:nv_m+nv_a] .= v_a
            arrays.t1sx .= arrays.x
            arrays.t1sF .= 0.0
            # Views
            varx = view(arrays.x, arrays.map)
            t1svarx = view(arrays.t1sx, arrays.map)
        end
    end
    @timeit timer "Seeding" begin
        if isa(device, CUDADevice)
            CUDA.@sync begin
                @cuda threads=nthreads blocks=nblocks myseed_kernel_gpu(
                    t1svarx,
                    varx,
                    arrays.t1sseeds,
                )
            end
        else
            myseed_kernel_cpu(
                t1svarx,
                varx,
                arrays.t1sseeds,
            )
        end
    end
    nthreads = 256
    nblocks = ceil(Int64, nbus/nthreads)

    @timeit timer "Function" begin
        residualFunction_polar!(
            arrays.t1sF,
            arrays.t1sx[1:nv_m],
            arrays.t1sx[nv_m+1:nv_m+nv_a],
            ybus_re, ybus_im,
            pinj, qinj,
            pv, pq, nbus
        )
    end

    @timeit timer "Get partials" begin
        if isa(device, CUDADevice)
            CUDA.@sync begin
                @cuda threads=nthreads blocks=nblocks getpartials_gpu(
                    arrays.compressedJ,
                    arrays.t1sF
                )
            end
        else
            ev = getpartials_cpu(
                arrays.compressedJ,
                arrays.t1sF,
            )
        end
    end
    @timeit timer "Uncompress" begin
        # Uncompress matrix. Sparse matrix elements have different names with CUDA
        if arrays.J isa SparseArrays.SparseMatrixCSC
            @inbounds for i in 1:nmap
                for j in arrays.J.colptr[i]:arrays.J.colptr[i+1]-1
                    @inbounds arrays.J.nzval[j] = arrays.compressedJ[arrays.coloring[i],arrays.J.rowval[j]]
                end
            end
        end
        if arrays.J isa CUDA.CUSPARSE.CuSparseMatrixCSR
            CUDA.@sync begin
                @cuda threads=nthreads blocks=nblocks uncompress(
                        arrays.J.nzVal,
                        arrays.J.rowPtr,
                        arrays.J.colVal,
                        arrays.compressedJ,
                        arrays.coloring, nmap
                )
            end
        end
        return nothing
    end
end

function residualJacobianAD!(arrays::DesignJacobianAD, residualFunction_polar!, v_m, v_a,
                             ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus, timer = nothing)
    device = isa(arrays.J, SparseArrays.SparseMatrixCSC) ? CPU() : CUDADevice()

    @timeit timer "Before" begin
        @timeit timer "Setup" begin
            npinj = size(pinj , 1)
            nv_m = size(v_m, 1)
            nmap = size(arrays.map, 1)
            nthreads=256
            nblocks=ceil(Int64, nmap/nthreads)
            n = npinj + nv_m
        end
        @timeit timer "Arrays" begin
            arrays.x[1:nv_m] .= v_m
            arrays.x[nv_m+1:nv_m+npinj] .= pinj
            arrays.t1sx .= arrays.x
            arrays.t1sF .= 0.0
            # Views
            varx = view(arrays.x, arrays.map)
            t1svarx = view(arrays.t1sx, arrays.map)
        end
    end
    @timeit timer "Seeding" begin
        if isa(device, CUDADevice)
            CUDA.@sync begin
                @cuda threads=nthreads blocks=nblocks myseed_kernel_gpu(
                    t1svarx,
                    varx,
                    arrays.t1sseeds,
                )
            end
        else
            ev = myseed_kernel_cpu(
                t1svarx,
                varx,
                arrays.t1sseeds,
            )
        end
    end
    nthreads=256
    nblocks=ceil(Int64, nbus/nthreads)
    @timeit timer "Function" begin
        residualFunction_polar!(
            arrays.t1sF,
            arrays.t1sx[1:nv_m],
            v_a,
            ybus_re, ybus_im,
            arrays.t1sx[nv_m+1:nv_m + npinj], qinj,
            pv, pq, nbus
        )
    end

    @timeit timer "Get partials" begin
        if isa(device, CUDADevice)
            CUDA.@sync begin
                @cuda threads=nthreads blocks=nblocks getpartials_gpu(
                    arrays.compressedJ,
                    arrays.t1sF
                )
            end
        else
            ev = getpartials_cpu(
                arrays.compressedJ,
                arrays.t1sF,
            )
        end
    end
    @timeit timer "Uncompress" begin
        # Uncompress matrix. Sparse matrix elements have different names with CUDA
        if arrays.J isa SparseArrays.SparseMatrixCSC
            for i in 1:nmap
                for j in arrays.J.colptr[i]:arrays.J.colptr[i+1]-1
                    @inbounds arrays.J.nzval[j] = arrays.compressedJ[arrays.coloring[i],arrays.J.rowval[j]]
                end
            end
        end
        if arrays.J isa CUDA.CUSPARSE.CuSparseMatrixCSR
            CUDA.@sync begin
                @cuda threads=nthreads blocks=nblocks uncompress(
                        arrays.J.nzVal,
                        arrays.J.rowPtr,
                        arrays.J.colVal,
                        arrays.compressedJ,
                        arrays.coloring, nmap
                )
            end
        end
        return nothing
    end
end

function Base.show(io::IO, ad::AbstractJacobianAD)
    ncolor = size(unique(ad.coloring), 1)
    print(io, "Number of Jacobian colors: ", ncolor)
end

end
