module AD

using CUDA
using ForwardDiff
using KernelAbstractions
using SparseArrays
using TimerOutputs


@kernel function myseed_kernel(duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
                 seeds::AbstractArray{ForwardDiff.Partials{N,V}}) where {T,V,N}
    i = @index(Global, Linear)
    @inbounds duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
end

@kernel function getpartials_kernel(compressedJ, t1sF)
    i = @index(Global, Linear)
    @inbounds compressedJ[:,i] .= ForwardDiff.partials.(t1sF[i]).values
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
                             ybus_re, ybus_im, pinj, qinj, pv, pq, nbus, to = nothing)
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
            arrays.t1sx .= arrays.x
            arrays.x[nv_m+1:nv_m+nv_a] .= v_a
            arrays.t1sF .= 0.0
        end
    end
    @timeit to "Seeding" begin
        ev = myseed_kernel(device)(
            arrays.t1svarx,
            arrays.varx,
            arrays.t1sseeds,
            ndrange=length(arrays.t1svarx))
        wait(ev)
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
        ev = getpartials_kernel(device)(
            arrays.compressedJ,
            arrays.t1sF,
            ndrange=length(arrays.t1sF)
        )
        wait(ev)
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
                        arrays.coloring, nmap)
            end
        end
        return nothing
    end
end

struct JacobianAD
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
    function JacobianAD(J, coloring, F, v_m, v_a, pv, pq)
        nv_m = size(v_m, 1)
        nv_a = size(v_a, 1)
        n = nv_m + nv_a
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

end
