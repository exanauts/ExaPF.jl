import LinearAlgebra: ldiv!, \, *, mul!
using CUDAKernels
using SparseMatricesCSR
"""
    AbstractPreconditioner

Preconditioners for the iterative solvers mostly focused on GPUs

"""
abstract type AbstractPreconditioner end

"""
      overlap(Graph, subset, level)
Given subset embedded within Graph, compute subset2 such that
subset2 contains subset and all of its adjacent vertices.
"""
function overlap(Graph, subset; level=1)

  @assert level > 0
  subset2 = [LightGraphs.neighbors(Graph, v) for v in subset]
  subset2 = reduce(vcat, subset2)
  subset2 = unique(vcat(subset, subset2))

  level -= 1
  if level == 0
    return subset2
  else
    return overlap(Graph, subset2, level=level)
  end
end

function count_nnz(partitions, lpartitions, colptr, rowval, nzval, b)
    cnt = 0
    for k in 1:lpartitions[b]
        i = partitions[k, b]
        for col_ptr in colptr[i]:(colptr[i + 1] - 1)
            col = rowval[col_ptr]
            ptr = findfirst(x -> x==col, partitions[:, b])
            if typeof(ptr) != Nothing
                cnt += 1
            end
        end
    end
    return cnt
end

function create_nnz_indexes(nnz_idx, blk_idx, partitions, lpartitions,
                            rowPtr, colVal, nzVal, blocksize, npart)
    cnt = 0
    for b in 1:npart
        for k in 1:lpartitions[b]
            # select row
            i = partitions[k, b]
            # iterate matrix
            for row_ptr in rowPtr[i]:(rowPtr[i + 1] - 1)
                # retrieve column value
                col = colVal[row_ptr]
                # iterate partition list and see if pertains to it
                for j in 1:lpartitions[b]
                    if col == partitions[j, b]
                        cnt += 1
                        nnz_idx[cnt] = row_ptr
                        blk_idx[cnt] = k + (j - 1)*blocksize + (b - 1)*blocksize*blocksize
                    end
                end
            end
        end
    end
end

"""
    BlockJacobiPreconditioner

Creates an object for the block-Jacobi preconditioner

* `nblocks::Int64`: Number of partitions or blocks.
* `blocksize::Int64`: Size of each block.
* `nJs::Int64`: Size of the blocks. For the GPUs these all have to be of equal size.
* `partitions::Vector{Vector{Int64}}``: `npart` partitions stored as lists
* `cupartitions`: `partitions` transfered to the GPU
* `lpartitions::Vector{Int64}``: Length of each partitions.
* `culpartitions::Vector{Int64}``: Length of each partitions, on the GPU.
* `blocks`: Dense blocks of the block-Jacobi
* `cublocks`: `Js` transfered to the GPU
* `map`: The partitions as a mapping to construct views
* `cumap`: `cumap` transferred to the GPU`
* `part`: Partitioning as output by Metis
* `cupart`: `part` transferred to the GPU
* `P`: The sparse precondition matrix whose values are updated at each iteration
"""
struct BlockJacobiPreconditioner{AT,GAT,VI,GVI,MT,GMT,MI,GMI,SMT,VF,GVF} <: AbstractPreconditioner
    nblocks::Int64
    blocksize::Int64
    partitions::MI
    cupartitions::Union{GMI,Nothing}
    lpartitions::VI
    culpartitions::Union{GVI,Nothing}
    rest_size::VI
    curest_size::Union{GVI,Nothing}
    blocks::AT
    cublocks::Union{GAT,Nothing}
    map::VI
    cumap::Union{GVI,Nothing}
    part::VI
    cupart::Union{GVI,Nothing}
    id::Union{GMT,MT}
    yaux::VF
    cuyaux::Union{GVF,Nothing}
    # block extraction auxiliary structures
    nnz_idx::VI
    blk_idx::VI
    cunnz_idx::Union{GVI, Nothing}
    cublk_idx::Union{GVI, Nothing}
    function BlockJacobiPreconditioner(J, npart, device=CPU(), olevel=0) where {}
        if isa(device, CPU)
            AT  = Array{Float64,3}
            GAT = Nothing
            VI  = Vector{Int64}
            GVI = Nothing
            MT  = Matrix{Float64}
            GMT = Nothing
            MI  = Matrix{Int64}
            GMI = Nothing
            SMT = SparseMatrixCSC{Float64,Int64}
            VF = Vector{Float64}
            GVF = Nothing
        elseif isa(device, GPU)
            AT  = Array{Float64,3}
            GAT = CuArray{Float64,3}
            VI  = Vector{Int64}
            GVI = CuVector{Int64}
            MT  = Matrix{Float64}
            GMT = CuMatrix{Float64}
            MI  = Matrix{Int64}
            GMI = CuMatrix{Int64}
            SMT = CUDA.CUSPARSE.CuSparseMatrixCSR{Float64}
            VF = Vector{Float64}
            GVF = CuVector{Float64}
            J = SparseMatrixCSC(J)
        else
            error("Unknown device type")
        end
        m, n = size(J)
        if npart < 2
            error("Number of partitions `npart` should be at" *
                  "least 2 for partitioning in Metis")
        end
        adj = build_adjmatrix(J)
        g = LightGraphs.Graph(adj)
        part = Metis.partition(g, npart)
        partitions = Vector{Vector{Int64}}()
        for i in 1:npart
            push!(partitions, [])
        end
        for (i,v) in enumerate(part)
            push!(partitions[v], i)
        end
        # We keep track of the partition size pre-overlap.
        # This will allow us to implement the RAS update.
        rest_size = VI(undef, npart)
        rest_size = length.(partitions)
        # overlap
        if olevel > 0
            for i in 1:npart
                partitions[i] = overlap(g, partitions[i], level=olevel)
            end
        end
        lpartitions = VI(undef, npart)
        lpartitions = length.(partitions)
        blocksize = maximum(length.(partitions))
        blocks = AT(undef, blocksize, blocksize, npart)
        # Get partitions into bit typed structure
        bpartitions = MI(undef, blocksize, npart)
        bpartitions .= 0.0
        for i in 1:npart
            bpartitions[1:length(partitions[i]),i] .= VI(partitions[i])
        end
        id = MT(I, blocksize, blocksize)
        for i in 1:npart
            blocks[:,:,i] .= id
        end
        nmap = 0
        for b in partitions
            nmap += length(b)
        end
        map = VI(undef, nmap)
        part = VI(undef, nmap)
        for b in 1:npart
            for (i,el) in enumerate(partitions[b])
                map[el] = i
                part[el] = b
            end
        end

        row = Vector{Float64}()
        col = Vector{Float64}()
        nzval = Vector{Float64}()

        for b in 1:npart
            for x in partitions[b]
                for y in partitions[b]
                    push!(row, x)
                    push!(col, y)
                    push!(nzval, 1.0)
                end
            end
        end
        yaux = VF(undef, n)

        # here we create data structures to accelerate the block extraction.
        # 1) For each partition, obtain number of nnz's in J
        
        ctr = 0
        for b in 1:npart
            ctr += count_nnz(bpartitions, lpartitions, J.colptr, J.rowval, J.nzval, b)
        end
        println("Total number of non-zeros in blocks: ", ctr)
        # 2) Create index vector that will track all NNZ
        nnz_idx = VI(undef, ctr)
        blk_idx = VI(undef, ctr)
        
        II, JJ, VV = findnz(J)
        JCSR = sparsecsr(II, JJ, VV)
        
        create_nnz_indexes(nnz_idx, blk_idx, bpartitions, lpartitions,
                            JCSR.rowptr, JCSR.colval, JCSR.nzval, blocksize, npart)
        if isa(device, GPU)
            id = GMT(I, blocksize, blocksize)
            cubpartitions = GMI(bpartitions)
            culpartitions = GVI(lpartitions)
            curest_size = GVI(rest_size)
            cublocks = GAT(blocks)
            cumap = CUDA.cu(map)
            cupart = CUDA.cu(part)
            cuyaux = GVF(undef, n)
            cunnz_idx = GVI(nnz_idx)
            cublk_idx = GVI(blk_idx)
        else
            cublocks = nothing
            cubpartitions = nothing
            cumap = nothing
            cupart = nothing
            id = MT(I, blocksize, blocksize)
            culpartitions = nothing
            curest_size = nothing
            cuyaux = nothing
            cunnz_idx = nothing
            cublk_idx = nothing
        end
        return new{AT,GAT,VI,GVI,MT,GMT,MI,GMI,SMT,VF,GVF}(npart, blocksize, bpartitions, cubpartitions, lpartitions, culpartitions, rest_size, curest_size, blocks, cublocks, map, cumap, part, cupart, id, yaux, cuyaux, nnz_idx, blk_idx, cunnz_idx, cublk_idx)
    end
end

Base.eltype(::BlockJacobiPreconditioner{AT,GAT,VI,GVI,MT,GMT,MI,GMI,SMT}) where {AT,GAT,VI,GVI,MT,GMT,MI,GMI,SMT} = Float64

@kernel function multiply_blocks_gpu!(y, b, p_len, rp_len, part, blocks)
    i = @index(Global, Linear)
    len = p_len[i]
    rlen = rp_len[i]
    idxA = -1
    idxB = -1
    accum = 0.0
    # homemade matrix multiply. This is probably not a good idea.
    @inbounds for j=1:rlen
        idxA = part[j, i]
        accum = 0.0
        @inbounds for k=1:len
            idxB = part[k, i]
            accum = accum + blocks[j, k, i]*b[idxB]
        end
        y[idxA] = accum
    end
end

@inline function (*)(C::BlockJacobiPreconditioner, b::Vector{Float64})
    n = size(b, 1)
    C.yaux .= 0.0
    for i=1:C.nblocks
        rlen = C.lpartitions[i]
        part = C.partitions[1:rlen, i]
        blck = C.blocks[1:rlen, 1:rlen, i]
        for j=1:C.rest_size[i]
            idx = part[j]
            C.yaux[idx] += dot(blck[j, :], b[part])
        end
    end
    return C.yaux
end

@inline function (*)(C::BlockJacobiPreconditioner, b::CuVector{Float64})
    n = size(b, 1)
    C.cuyaux .= 0.0
    mblock_gpu_kernel! = multiply_blocks_gpu!(CUDADevice())
    ev = mblock_gpu_kernel!(C.cuyaux, b, C.culpartitions, C.curest_size,
                            C.cupartitions, C.cublocks, ndrange=C.nblocks)
    wait(ev)
    return C.cuyaux
end

"""
    build_adjmatrix

Build the adjacency matrix of a matrix A corresponding to the undirected graph

"""
function build_adjmatrix(A)
    rows = Int64[]
    cols = Int64[]
    vals = Float64[]
    rowsA = rowvals(A)
    m, n = size(A)
    for i = 1:n
        for j in nzrange(A, i)
            push!(rows, rowsA[j])
            push!(cols, i)
            push!(vals, 1.0)

            push!(rows, i)
            push!(cols, rowsA[j])
            push!(vals, 1.0)
        end
    end
    return sparse(rows,cols,vals,size(A,1),size(A,2))
end

"""
    fillblock_gpu

Fill the dense blocks of the preconditioner from the sparse CSR matrix arrays

"""
@kernel function fillblock_gpu!(blocks, blocksize, partition, map, rowPtr, colVal, nzVal, part, lpartitions, id)
    b = @index(Global, Linear)
    @inbounds for k in 1:lpartitions[b]
        # select row
        i = partition[k, b]
        # iterate matrix
        for row_ptr in rowPtr[i]:(rowPtr[i + 1] - 1)
            # retrieve column value
            col = colVal[row_ptr]
            # iterate partition list and see if pertains to it
            for j in 1:lpartitions[b]
                if col == partition[j, b]
                    @inbounds blocks[k, j, b] = nzVal[row_ptr]
                end
            end
        end
    end
end

@kernel function fillblock_direct_gpu!(blocks, block_idx, nnz_idx, nzVal)
    i = @index(Global, Linear)
    blocks[block_idx[i]] = nzVal[nnz_idx[i]]
end

@kernel function reset_blocks!(blocks, blocksize, nblocks, id)
    b = @index(Global, Linear)
    for i in 1:blocksize
        for j in 1:blocksize
            blocks[i,j,b] = id[i,j]
        end
    end
end

function _update_gpu(p, j_rowptr, j_colval, j_nzval, device)
    nblocks = p.nblocks
    reset_blocks_kernel! = reset_blocks!(device)
    fillblock_gpu_kernel! = fillblock_gpu!(device)
    fillblock_direct_gpu_kernel! = fillblock_direct_gpu!(device)
    # Fill Block Jacobi" begin
    ev = reset_blocks_kernel!(p.cublocks, size(p.id,1), nblocks, p.id,
                              ndrange=nblocks, dependencies=Event(device))
    wait(ev)
    #ev = fillblock_direct_gpu_kernel!(p.cublocks, p.cublk_idx, p.cunnz_idx, j_nzval,
    #                                  ndrange=size(p.cublk_idx, 1), dependencies=Event(device))
    ev = fillblock_gpu_kernel!(p.cublocks, size(p.id,1), p.cupartitions, p.cumap, j_rowptr, j_colval, j_nzval, p.cupart, p.culpartitions, p.id, ndrange=nblocks, dependencies=Event(device))
    wait(ev)
    # Invert blocks begin
    blocklist = Array{CuArray{Float64,2}}(undef, nblocks)
    for b in 1:nblocks
        blocklist[b] = p.cublocks[:,:,b]
    end
    CUDA.@sync pivot, info = CUDA.CUBLAS.getrf_batched!(blocklist, true)
    CUDA.@sync pivot, info, blocklist = CUDA.CUBLAS.getri_batched(blocklist, pivot)
    for b in 1:nblocks
        p.cublocks[:,:,b] .= blocklist[b]
    end
end

"""
    function update(J::CuSparseMatrixCSR, p)

Update the preconditioner `p` from the sparse Jacobian `J` in CSR format for the GPU

1) The dense blocks `cuJs` are filled from the sparse Jacobian `J`
2) To a batch inversion of the dense blocks using CUBLAS
3) Extract the preconditioner matrix `p.P` from the dense blocks `cuJs`

"""
function update(p, J::CUSPARSE.CuSparseMatrixCSR, device)
    _update_gpu(p, J.rowPtr, J.colVal, J.nzVal, device)
end
function update(p, J::Transpose{T, CUSPARSE.CuSparseMatrixCSR{T}}, device) where T
    Jt = CUSPARSE.CuSparseMatrixCSC(J.parent)
    _update_gpu(p, Jt.colPtr, Jt.rowVal, Jt.nzVal, device)
end

@kernel function update_cpu_kernel!(colptr, rowval, nzval, p, lpartitions)
    nblocks = length(p.partitions)
    # Fill Block Jacobi
    b = @index(Global, Linear)
    blocksize = size(p.id,1)
    for i in 1:blocksize
        for j in 1:blocksize
            p.blocks[i,j,b] = p.id[i,j]
        end
    end
    for k in 1:lpartitions[b]
        i = p.partitions[k, b]
        for col_ptr in colptr[i]:(colptr[i + 1] - 1)
            col = rowval[col_ptr]
            ptr = findfirst(x -> x==col, p.partitions[:, b])
            if typeof(ptr) != Nothing
                # Here we assume the natural order of the partition
                # matches that of the block.
                @inbounds p.blocks[ptr, k, b] = nzval[col_ptr]
            end
        end
    end
    # Invert blocks
    p.blocks[:,:,b] .= inv(p.blocks[:,:,b])
end

"""
    function update(J::SparseMatrixCSC, p)

Update the preconditioner `p` from the sparse Jacobian `J` in CSC format for the CPU

Note that this implements the same algorithm as for the GPU and becomes very slow on CPU with growing number of blocks.

"""
function update(p, J::SparseMatrixCSC, device)
    kernel! = update_cpu_kernel!(device)
    ev = kernel!(J.colptr, J.rowval, J.nzval, p, p.lpartitions, ndrange=p.nblocks, dependencies=Event(device))
    wait(ev)
end
function update(p, J::Transpose{T, SparseMatrixCSC{T, I}}) where {T, I}
    ix, jx, zx = findnz(J.parent)
    update(p, sparse(jx, ix, zx))
end

is_valid(precond::BlockJacobiPreconditioner) = _check_nan(precond.P)
_check_nan(P::SparseMatrixCSC) = !any(isnan.(P.nzval))
_check_nan(P::CUSPARSE.CuSparseMatrixCSR) = !any(isnan.(P.nzVal))

function Base.show(precond::BlockJacobiPreconditioner)
    npartitions = precond.npart
    println("Block Jacobi block size: $(precond.nJs)")
end

