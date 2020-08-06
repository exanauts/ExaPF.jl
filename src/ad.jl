module AD

using CUDA
using ForwardDiff
using KernelAbstractions
using SparseArrays
using TimerOutputs

abstract type AbstractJacobianAD end

struct StateJacobianAD <: AbstractJacobianAD
    J
    compressedJ
    coloring
    t1sseeds
    t1sF
    x
    t1sx
    varx
    t1svarx
    map
    function StateJacobianAD(J, coloring, F, v_m, v_a, pbus, pv, pq, ref)
        nv_m = size(v_m, 1)
        nv_a = size(v_a, 1)
        ncolor = size(unique(coloring),1)
        if F isa Array
            T = Vector
            M = Matrix
            A = Array
        elseif F isa CuArray
            T = CuVector
            M = CuMatrix
            A = CuArray
        else
            error("Wrong array type ", typeof(F))
        end

        mappv = [i + nv_m for i in pv]
        mappq = [i + nv_m for i in pq]
        map = T{Int64}(vcat(mappv, mappq, pq))
        nmap = size(map,1)

        t1s{N} =  ForwardDiff.Dual{Nothing,Float64, N} where N
        # x = T{Float64}(undef, nv_m + nv_a)
        x = T(zeros(Float64, nv_m + nv_a))
        t1sx = T{t1s{ncolor}}(x)
        # t1sF = T{t1s{ncolor}}(undef, nmap)
        t1sF = T{t1s{ncolor}}(zeros(Float64, nmap))
        varx = view(x,map)
        t1sseedvec = zeros(Float64, ncolor)
        t1sseeds = T{ForwardDiff.Partials{ncolor,Float64}}(undef, nmap)
        for i in 1:nmap
            for j in 1:ncolor
                if coloring[i] == j
                    t1sseedvec[j] = 1.0
                end
            end
            t1sseeds[i] = ForwardDiff.Partials{ncolor, Float64}(NTuple{ncolor, Float64}(t1sseedvec))
            t1sseedvec .= 0
        end
        compressedJ = M{Float64}(zeros(Float64, ncolor, nmap))
        t1svarx = view(t1sx, map)
        nthreads=256
        nblocks=ceil(Int64, nmap/nthreads)
        return new(J, compressedJ, coloring, t1sseeds, t1sF, x, t1sx, varx, t1svarx, map)
    end
end

struct DesignJacobianAD <: AbstractJacobianAD
    J
    compressedJ
    coloring
    t1sseeds
    t1sF
    x
    t1sx
    varx
    t1svarx
    map
    function DesignJacobianAD(J, coloring, F, v_m, v_a, pbus, pv, pq, ref)
        nv_m = size(v_m, 1)
        nv_a = size(v_a, 1)
        npbus = size(pbus, 1)
        nref = size(ref, 1)
        # ncolor = size(unique(coloring),1)
        if F isa Array
            T = Vector
            M = Matrix
            A = Array
        elseif F isa CuArray
            T = CuVector
            M = CuMatrix
            A = CuArray
        else
            error("Wrong array type ", typeof(F))
        end

        mappv =  [i + nv_a for i in pv]
        map = T{Int64}(vcat(ref, mappv, pv))
        nmap = size(map,1)

        t1s{N} =  ForwardDiff.Dual{Nothing,Float64, N} where N
        # x = T{Float64}(undef, nv_m + nv_a)
        x = T(zeros(Float64, npbus + nv_a))
        ncolor = length(x)
        t1sx = T{t1s{ncolor}}(x)
        # t1sF = T{t1s{ncolor}}(undef, nmap)
        t1sF = T{t1s{ncolor}}(zeros(Float64, length(F)))
        varx = view(x,map)
        t1sseedvec = zeros(Float64, ncolor)
        t1sseeds = T{ForwardDiff.Partials{ncolor,Float64}}(undef, nmap)
        for i in 1:nmap
            for j in 1:ncolor
                # if coloring[i] == j
                    t1sseedvec[j] = 1.0
                # end
            end
            t1sseeds[i] = ForwardDiff.Partials{ncolor, Float64}(NTuple{ncolor, Float64}(t1sseedvec))
            t1sseedvec .= 0
        end
        compressedJ = M{Float64}(zeros(Float64, ncolor, length(F)))
        t1svarx = view(t1sx, map)
        nthreads=256
        nblocks=ceil(Int64, nmap/nthreads)
        return new(J, compressedJ, coloring, t1sseeds, t1sF, x, t1sx, varx, t1svarx, map)
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

function residualJacobianAD!(arrays, residualFunction_polar!, v_m, v_a,
                             ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus, to = nothing)
    device = isa(arrays.J, SparseArrays.SparseMatrixCSC) ? CPU() : CUDADevice()
    @timeit to "Before" begin
        @timeit to "Setup" begin
            nv_m = size(v_m, 1)
            nv_a = size(v_a, 1)
            nmap = size(arrays.map, 1)
            nthreads=256
            nblocks=ceil(Int64, nmap/nthreads)
            n = nv_m + nv_a
        end
        @timeit to "Arrays" begin
            arrays.x[1:nv_m] .= v_m
            arrays.x[nv_m+1:nv_m+nv_a] .= v_a
            arrays.t1sx .= arrays.x
            arrays.t1sF .= 0.0
        end
    end
    @timeit to "Seeding" begin
        if isa(device, CUDADevice)
            CUDA.@sync begin
                @cuda threads=nthreads blocks=nblocks myseed_kernel_gpu(
                    arrays.t1svarx,
                    arrays.varx,
                    arrays.t1sseeds,
                )
            end
        else
            ev = myseed_kernel_cpu(
                arrays.t1svarx,
                arrays.varx,
                arrays.t1sseeds,
            )
        end
    end
    nthreads=256
    nblocks=ceil(Int64, nbus/nthreads)

    @timeit to "Function" begin
        residualFunction_polar!(
            arrays.t1sF,
            arrays.t1sx[1:nv_m],
            arrays.t1sx[nv_m+1:nv_m+nv_a],
            ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
            ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
            pinj, qinj,
            pv, pq, nbus
        )
    end

    @timeit to "Get partials" begin
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
    @timeit to "Uncompress" begin
        # Uncompress matrix. Sparse matrix elements have different names with CUDA
        if arrays.J isa SparseArrays.SparseMatrixCSC
            for i in 1:nmap
                for j in arrays.J.colptr[i]:arrays.J.colptr[i+1]-1
                    arrays.J.nzval[j] = arrays.compressedJ[arrays.coloring[i],arrays.J.rowval[j]]
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

function designJacobianAD!(arrays, residualFunction_polar!, v_m, v_a,
                             ybus_re, ybus_im, pinj, qinj, pv, pq, ref, nbus, to = nothing)
    device = isa(arrays.J, SparseArrays.SparseMatrixCSC) ? CPU() : CUDADevice()

    @timeit to "Before" begin
        @timeit to "Setup" begin
            npinj = size(pinj , 1)
            nv_a = size(v_a, 1)
            nmap = size(arrays.map, 1)
            nthreads=256
            nblocks=ceil(Int64, nmap/nthreads)
            n = npinj + nv_a
        end
        @timeit to "Arrays" begin
            arrays.x[1:nv_a] .= v_a
            arrays.x[nv_a+1:nv_a+npinj] .= pinj
            arrays.t1sx .= arrays.x
            arrays.t1sF .= 0.0
        end
    end
    @timeit to "Seeding" begin
        if isa(device, CUDADevice)
            CUDA.@sync begin
                @cuda threads=nthreads blocks=nblocks myseed_kernel_gpu(
                    arrays.t1svarx,
                    arrays.varx,
                    arrays.t1sseeds,
                )
            end
        else
            ev = myseed_kernel_cpu(
                arrays.t1svarx,
                arrays.varx,
                arrays.t1sseeds,
            )
        end
    end
    nthreads=256
    nblocks=ceil(Int64, nbus/nthreads)
    t1sv_m = isa(arrays.J, SparseArrays.SparseMatrixCSC) ? zeros(eltype(arrays.t1sF), length(v_m)) : CUDA.zeros(eltype(arrays.t1sF), length(v_m))
    t1sv_m .= v_m
    @timeit to "Function" begin
        residualFunction_polar!(
            arrays.t1sF,
            t1sv_m,
            arrays.t1sx[1:nv_a],
            ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
            ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
            arrays.t1sx[nv_a+1:nv_a + npinj], qinj,
            pv, pq, nbus
        )
    end

    @timeit to "Get partials" begin
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
    @timeit to "Uncompress" begin
        # Uncompress matrix. Sparse matrix elements have different names with CUDA
        if arrays.J isa SparseArrays.SparseMatrixCSC
            for i in 1:nmap
                for j in arrays.J.colptr[i]:arrays.J.colptr[i+1]-1
                    arrays.J.nzval[j] = arrays.compressedJ[arrays.coloring[i],arrays.J.rowval[j]]
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


end
