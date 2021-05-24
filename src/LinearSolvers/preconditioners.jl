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
  subset2 = sort(unique(vcat(subset, subset2)))

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
    blocks::AT
    cublocks::Union{GAT,Nothing}
    map::VI
    cumap::Union{GVI,Nothing}
    part::VI
    cupart::Union{GVI,Nothing}
    P::SMT
    id::Union{GMT,MT}
    yaux::VF
    cuyaux::Union{GVF,Nothing}
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
        P = sparse(row, col, nzval)
        yaux = VF(undef, n)
        if isa(device, GPU)
            id = GMT(I, blocksize, blocksize)
            cubpartitions = GMI(bpartitions)
            culpartitions = GVI(lpartitions)
            cublocks = GAT(blocks)
            cumap = CUDA.cu(map)
            cupart = CUDA.cu(part)
            P = SMT(P)
            cuyaux = GVF(undef, n)
        else
            cublocks = nothing
            cubpartitions = nothing
            cumap = nothing
            cupart = nothing
            id = MT(I, blocksize, blocksize)
            culpartitions = nothing
            cuyaux = nothing
        end
        return new{AT,GAT,VI,GVI,MT,GMT,MI,GMI,SMT,VF,GVF}(npart, blocksize, bpartitions, cubpartitions, lpartitions, culpartitions, blocks, cublocks, map, cumap, part, cupart, P, id, yaux, cuyaux)
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
Base.eltype(::BlockJacobiPreconditioner{AT,GAT,VI,GVI,MT,GMT,MI,GMI,SMT}) where {AT,GAT,VI,GVI,MT,GMT,MI,GMI,SMT} = Float64

@inline function mul!(y, C::BlockJacobiPreconditioner, b::Vector{Float64})
    n = size(b, 1)
    y .= 0.0
    for i=1:C.nblocks
        rlen = C.lpartitions[i]
        part = C.partitions[1:rlen, i]
        blck = C.blocks[1:rlen, 1:rlen, i]
        y[part] .+= blck*b[part]
    end
end

@inline function mul!(y, C::BlockJacobiPreconditioner, b::CuVector{Float64})
    n = size(b, 1)
    y .= 0.0
    for i=1:C.nblocks
        rlen = C.culpartitions[i]
        part = C.cupartitions[1:rlen, i]
        blck = C.cublocks[1:rlen, 1:rlen, i]
        y[part] += blck*b[part]
    end
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

    @inbounds for i in 1:lpartitions[b]
        @inbounds for j in rowPtr[partition[i,b]]:rowPtr[partition[i,b]+1]-1
            if b == part[colVal[j]]
                @inbounds blocks[map[partition[i,b]], map[colVal[j]], b] = nzVal[j]
            end
        end
    end
end

"""
    fillP_gpu

Update the values of the preconditioner matrix from the dense Jacobi blocks

"""
@kernel function fillP_gpu!(blocks, partition, map, rowPtr, colVal, nzVal, part, lpartitions)
    b = @index(Global, Linear)
    @inbounds for i in 1:lpartitions[b]
        @inbounds for j in rowPtr[partition[i,b]]:rowPtr[partition[i,b]+1]-1
            if b == part[colVal[j]]
                @inbounds nzVal[j] += blocks[map[partition[i,b]], map[colVal[j]],b]
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
    p.P.nzVal .= 0.0
    # Move blocks to P" begin
    ev = fillP_gpu_kernel!(p.cublocks, p.cupartitions, p.cumap, p.P.rowPtr, p.P.colVal, p.P.nzVal, p.cupart, p.culpartitions, ndrange=nblocks, dependencies=Event(device))
    wait(ev)
    return p.P
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
    #for k in 1:lpartitions[b]
    #    i = p.partitions[k,b]
    #    println(i)
    #    for j in colptr[i]:colptr[i+1]-1
    #        if b == p.part[rowval[j]]
    #            @inbounds p.blocks[p.map[rowval[j]], p.map[i], b] = nzval[j]
    #        end
    #    end
    #end
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
    # Move blocks to P
    for k in 1:lpartitions[b]
        i = p.partitions[k,b]
        for j in p.P.colptr[i]:p.P.colptr[i+1]-1
            if b == p.part[p.P.rowval[j]]
                @inbounds p.P.nzval[j] += p.blocks[p.map[p.P.rowval[j]], p.map[i], b]
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
    p.P.nzval .= 0.0
    # Fill Block Jacobi
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
    # Move blocks to P
    for b in 1:p.nblocks
        for k in 1:p.lpartitions[b]
            i = p.partitions[k,b]
            for j in p.P.colptr[i]:p.P.colptr[i+1]-1
                if b == p.part[p.P.rowval[j]]
                    p.P.nzval[j] += p.blocks[p.map[p.P.rowval[j]], p.map[i], b]
                end
            end
        end
    end
    return p.P
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
    nblock = div(size(precond.P, 1), npartitions)
    println("#partitions: $npartitions, Blocksize: n = ", nblock,
            " Mbytes = ", (nblock*nblock*npartitions*8.0)/(1024.0*1024.0))
    println("Block Jacobi block size: $(precond.nJs)")
end

