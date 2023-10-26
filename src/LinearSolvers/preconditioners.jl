import LinearAlgebra: ldiv!, \, *, mul!

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

"""
    BlockJacobiPreconditioner

Overlapping-Schwarz preconditioner.

### Attributes

* `nblocks::Int64`: Number of partitions or blocks.
* `blocksize::Int64`: Size of each block.
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
"""
struct BlockJacobiPreconditioner{AT,GAT,VI,GVI,GMT,MI,GMI} <: AbstractPreconditioner
    nblocks::Int64
    blocksize::Int64
    partitions::MI
    cupartitions::GMI
    lpartitions::VI
    culpartitions::GVI
    rest_size::VI
    curest_size::GVI
    blocks::AT
    cublocks::GAT
    map::VI
    cumap::GVI
    part::VI
    cupart::GVI
    id::GMT
end

function BlockJacobiPreconditioner(J, npart, device=CPU(), olevel=0) where {}
    if npart < 2
        error("Number of partitions `npart` should be at" *
                "least 2 for partitioning in Metis")
    end
    adj = build_adjmatrix(SparseMatrixCSC(J))
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
    rest_size = length.(partitions)
    # overlap
    if olevel > 0
        for i in 1:npart
            partitions[i] = overlap(g, partitions[i], level=olevel)
        end
    end
    lpartitions = length.(partitions)
    blocksize = maximum(length.(partitions))
    blocks = zeros(Float64, blocksize, blocksize, npart)
    # Get partitions into bit typed structure
    bpartitions = zeros(Int64, blocksize, npart)
    bpartitions .= 0.0
    for i in 1:npart
        bpartitions[1:length(partitions[i]),i] .= Vector{Int64}(partitions[i])
    end
    id = Matrix{Float64}(I, blocksize, blocksize)
    for i in 1:npart
        blocks[:,:,i] .= id
    end
    nmap = 0
    for b in partitions
        nmap += length(b)
    end
    map = zeros(Int64, nmap)
    part = zeros(Int64, nmap)
    for b in 1:npart
        for (i,el) in enumerate(partitions[b])
            map[el] = i
            part[el] = b
        end
    end

    id = adapt(device, id)
    cubpartitions = adapt(device, bpartitions)
    culpartitions = adapt(device, lpartitions)
    curest_size = adapt(device, rest_size)
    cublocks = adapt(device, blocks)
    cumap = adapt(device, map)
    cupart = adapt(device, part)
    return BlockJacobiPreconditioner(
        npart, blocksize, bpartitions,
        cubpartitions, lpartitions,
        culpartitions, rest_size,
        curest_size, blocks,
        cublocks, map,
        cumap, part,
        cupart, id
    )
end

function BlockJacobiPreconditioner(J::SparseMatrixCSC; nblocks=-1, device=CPU(), noverlaps=0)
    n = size(J, 1)
    npartitions = if nblocks > 0
        nblocks
    else
        div(n, 32)
    end
    if npartitions < 2
        npartitions = 2
    end
    return BlockJacobiPreconditioner(J, npartitions, device, noverlaps)
end

Base.eltype(::BlockJacobiPreconditioner) = Float64

# NOTE: Custom kernel to implement blocks - vector multiplication.
# The blocks have very unbalanced sizes, leading to imbalances
# between the different threads.
# CUBLAS.gemm_strided_batched has been tested has well, but is
# overall 3x slower than this custom kernel : due to the various sizes
# of the blocks, gemm_strided is performing too many unecessary operations,
# impairing its performance.
@kernel function mblock_kernel!(y, b, p_len, rp_len, part, blocks)
    p = size(b, 2)
    i, j = @index(Global, NTuple)
    len = p_len[i]
    rlen = rp_len[i]

    if j <= rlen
        for ℓ=1:p
            accum = 0.0
            idxA = @inbounds part[j, i]
            for k=1:len
                idxB = @inbounds part[k, i]
                @inbounds accum = accum + blocks[j, k, i]*b[idxB,ℓ]
            end
            y[idxA,ℓ] = accum
        end
    end
end

function mul!(y, C::BlockJacobiPreconditioner, b::Vector{T}) where T
    n = size(b, 1)
    fill!(y, zero(T))
    for i=1:C.nblocks
        rlen = C.lpartitions[i]
        part = C.partitions[1:rlen, i]
        blck = C.blocks[1:rlen, 1:rlen, i]
        for j=1:C.rest_size[i]
            idx = part[j]
            y[idx] += dot(blck[j, :], b[part])
        end
    end
end

function mul!(Y, C::BlockJacobiPreconditioner, B::Matrix{T}) where T
    n, p = size(B)
    fill!(Y, zero(T))
    for i=1:C.nblocks
        rlen = C.lpartitions[i]
        part = C.partitions[1:rlen, i]
        blck = C.blocks[1:rlen, 1:rlen, i]
        for rhs=1:p
            for j=1:C.rest_size[i]
                idx = part[j]
                Y[idx,rhs] += dot(blck[j, :], B[part,rhs])
            end
        end
    end
end

function mul!(y, C::BlockJacobiPreconditioner, b::AbstractVector{T}) where T
    device = KA.get_backend(b)
    n = size(b, 1)
    fill!(y, zero(T))
    max_rlen = maximum(C.rest_size)
    ndrange = (C.nblocks, max_rlen)
    mblock_kernel!(device)(
        y, b, C.culpartitions, C.curest_size,
        C.cupartitions, C.cublocks,
        ndrange=ndrange,
    )
    KA.synchronize(device)
end

function mul!(Y, C::BlockJacobiPreconditioner, B::AbstractMatrix{T}) where T
    device = KA.get_backend(B)
    n, p = size(B)
    fill!(Y, zero(T))
    max_rlen = maximum(C.rest_size)
    ndrange = (C.nblocks, max_rlen)
    mblock_kernel!(device)(
        Y, B, C.culpartitions, C.curest_size,
        C.cupartitions, C.cublocks,
        ndrange=ndrange,
    )
    KA.synchronize(device)
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
    _fillblock_gpu

Fill the dense blocks of the preconditioner from the sparse CSR matrix arrays

"""
@kernel function _fillblock_gpu!(blocks, blocksize, partition, map, rowPtr, colVal, nzVal, part, lpartitions, id)
    b = @index(Global, Linear)
    for i in 1:blocksize
        for j in 1:blocksize
            blocks[i,j,b] = id[i,j]
        end
    end

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

"""
    function update(J::SparseMatrixCSC, p)

Update the preconditioner `p` from the sparse Jacobian `J` in CSC format for the CPU

Note that this implements the same algorithm as for the GPU and becomes very slow on CPU with growing number of blocks.

"""
function update(p, J::SparseMatrixCSC, device)
    # TODO: Enabling threading leads to a crash here
    for b in 1:p.nblocks
        p.blocks[:,:,b] = p.id[:,:]
        for k in 1:p.lpartitions[b]
            i = p.partitions[k,b]
            for j in J.colptr[i]:J.colptr[i+1]-1
                if b == p.part[J.rowval[j]]
                    p.blocks[p.map[J.rowval[j]], p.map[i], b] = J.nzval[j]
                end
            end
        end
    end
    for b in 1:p.nblocks
        # Invert blocks
        p.blocks[:,:,b] .= inv(p.blocks[:,:,b])
    end
end

function Base.show(precond::BlockJacobiPreconditioner)
    npartitions = precond.npart
    nblock = precond.nblocks
    println("#partitions: $npartitions, Blocksize: n = ", nblock,
            " Mbytes = ", (nblock*nblock*npartitions*8.0)/(1024.0*1024.0))
    println("Block Jacobi block size: $(precond.nJs)")
end
