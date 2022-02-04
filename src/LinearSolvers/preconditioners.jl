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
            GAT = CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
            VI  = Vector{Int64}
            GVI = CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}
            MT  = Matrix{Float64}
            GMT = CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}
            MI  = Matrix{Int64}
            GMI = CuArray{Int64, 2, CUDA.Mem.DeviceBuffer}
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
        rest_size = length.(partitions)
        # overlap
        if olevel > 0
            for i in 1:npart
                partitions[i] = overlap(g, partitions[i], level=olevel)
            end
        end
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

        if isa(device, GPU)
            id = GMT(I, blocksize, blocksize)
            cubpartitions = GMI(bpartitions)
            culpartitions = GVI(lpartitions)
            curest_size = GVI(rest_size)
            cublocks = GAT(blocks)
            cumap = CUDA.cu(map)
            cupart = CUDA.cu(part)
        else
            cublocks = nothing
            cubpartitions = nothing
            cumap = nothing
            cupart = nothing
            id = MT(I, blocksize, blocksize)
            culpartitions = nothing
            curest_size = nothing
        end
        return new{AT,GAT,VI,GVI,MT,GMT,MI,GMI,SMT,VF,GVF}(npart, blocksize, bpartitions, cubpartitions, lpartitions, culpartitions, rest_size, curest_size, blocks, cublocks, map, cumap, part, cupart, id)
    end
end

function BlockJacobiPreconditioner(J::SparseMatrixCSC; nblocks=-1, device=CPU())
    n = size(J, 1)
    npartitions = if nblocks > 0
        nblocks
    else
        div(n, 32)
    end
    return BlockJacobiPreconditioner(J, npartitions, device)
end
BlockJacobiPreconditioner(J::CUSPARSE.CuSparseMatrixCSR; options...) = BlockJacobiPreconditioner(SparseMatrixCSC(J); options...)

Base.eltype(::BlockJacobiPreconditioner) = Float64

# NOTE: Custom kernel to implement blocks - vector multiplication.
# The blocks have very unbalanced sizes, leading to imbalances
# between the different threads.
# CUBLAS.gemm_strided_batched has been tested has well, but is
# overall 3x slower than this custom kernel : due to the various sizes
# of the blocks, gemm_strided is performing too many unecessary operations,
# impairing its performance.
@kernel function mblock_kernel!(y, b, p_len, rp_len, part, blocks)
    i, j = @index(Global, NTuple)
    len = p_len[i]
    rlen = rp_len[i]

    if j <= rlen
        accum = 0.0
        idxA = @inbounds part[j, i]
        for k=1:len
            idxB = @inbounds part[k, i]
            @inbounds accum = accum + blocks[j, k, i]*b[idxB]
        end
        y[idxA] = accum
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

function mul!(y, C::BlockJacobiPreconditioner, b::CuVector{T}) where T
    n = size(b, 1)
    fill!(y, zero(T))
    max_rlen = maximum(C.rest_size)
    ndrange = (C.nblocks, max_rlen)
    ev = mblock_kernel!(CUDADevice())(
        y, b, C.culpartitions, C.curest_size,
        C.cupartitions, C.cublocks, ndrange=ndrange,
    )
    wait(ev)
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

function _update_gpu(p, j_rowptr, j_colval, j_nzval, device)
    nblocks = p.nblocks
    fillblock_gpu_kernel! = fillblock_gpu!(device)
    fillP_gpu_kernel! = fillP_gpu!(device)
    # Fill Block Jacobi" begin
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
    return
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

