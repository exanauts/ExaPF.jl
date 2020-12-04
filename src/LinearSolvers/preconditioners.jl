
using CUDA
using CUDA.CUSPARSE
using KernelAbstractions
using LightGraphs
using LinearAlgebra
using Metis
using SparseArrays
using TimerOutputs


"""
    AbstractPreconditioner

Preconditioners for the iterative solvers mostly focused on GPUs

"""
abstract type AbstractPreconditioner end

"""
    BlockJacobiPreconditioner

Creates an object for the block-Jacobi preconditioner

* `npart::Int64`: Number of partitions or blocks
* `nJs::Int64`: Size of the blocks. For the GPUs these all have to be of equal size.
* `partitions::Vector{Vector{Int64}}``: `npart` partitions stored as lists
* `cupartitions`: `partitions` transfered to the GPU
* `Js`: Dense blocks of the block-Jacobi
* `cuJs`: `Js` transfered to the GPU
* `map`: The partitions as a mapping to construct views
* `cumap`: `cumap` transferred to the GPU`
* `part`: Partitioning as output by Metis
* `cupart`: `part` transferred to the GPU
* `P`: The sparse precondition matrix whose values are updated at each iteration
"""
mutable struct BlockJacobiPreconditioner <: AbstractPreconditioner
    npart::Int64
    nJs::Int64
    partitions::Vector{Vector{Int64}}
    cupartitions
    Js
    cuJs
    map
    cumap
    part
    cupart
    P
    id
    function BlockJacobiPreconditioner(J, npart, device=CPU())
        if isa(J, CuSparseMatrixCSR)
            J = SparseMatrixCSC(J)
        end
        m, n = size(J)
        if npart < 2
            error("Number of partitions `npart` should be at" *
                  "least 2 for partitioning in Metis")
        end
        adj = build_adjmatrix(J)
        g = Graph(adj)
        part = Metis.partition(g, npart)
        partitions = Vector{Vector{Int64}}()
        for i in 1:npart
            push!(partitions, [])
        end
        for (i,v) in enumerate(part)
            push!(partitions[v], i)
        end
        Js = Vector{Matrix{Float64}}(undef, npart)
        nJs = maximum(length.(partitions))
        id = Matrix{Float64}(I, nJs, nJs)
        for i in 1:npart
            Js[i] = Matrix{Float64}(I, nJs, nJs)
        end
        nmap = 0
        for b in partitions
            nmap += length(b)
        end
        map = Vector{Int64}(undef, nmap)
        part = Vector{Int64}(undef, nmap)
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
        if isa(device, CUDADevice)
            id = CuMatrix{Float64}(I, nJs, nJs)
            cupartitions = Vector{CuVector{Int64}}(undef, npart)
            for i in 1:npart
                cupartitions[i] = CuVector{Int64}(partitions[i])
            end
            cuJs = Vector{CuMatrix{Float64}}(undef, length(partitions))
            for i in 1:length(partitions)
                cuJs[i] = CuMatrix{Float64}(I, nJs, nJs)
            end
            cumap = cu(map)
            cupart = cu(part)
            P = CuSparseMatrixCSR(P)
        else
            cuJs = nothing
            cupartitions = nothing
            cumap = nothing
            cupart = nothing
            id = nothing
        end
        return new(npart, nJs, partitions, cupartitions, Js, cuJs, map, cumap, part, cupart, P, id)
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

Fill the dense blocks of the preconditioner from the sparse CSC matrix arrays

"""
function fillblock_gpu!(cuJs, partition, map, rowPtr, colVal, nzVal, part, b)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:length(partition)
        for j in rowPtr[partition[i]]:rowPtr[partition[i]+1]-1
            if b == part[colVal[j]]
                @inbounds cuJs[map[partition[i]], map[colVal[j]]] = nzVal[j]
            end
        end
    end
    return nothing
end

"""
    fillblock_gpu

Update the values of the preconditioner matrix from the dense Jacobi blocks

"""
function fillP_gpu!(cuJs, partition, map, rowPtr, colVal, nzVal, part, b)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:length(partition)
        for j in rowPtr[partition[i]]:rowPtr[partition[i]+1]-1
            if b == part[colVal[j]]
                @inbounds nzVal[j] += cuJs[map[partition[i]], map[colVal[j]]]
            end
        end
    end
    return nothing
end

function _update_gpu(j_rowptr, j_colval, j_nzval, p)
    nblocks = length(p.partitions)
    for el in p.cuJs
        el .= p.id
    end
    # Fill Block Jacobi" begin
    CUDA.@sync begin
        for b in 1:nblocks
            @cuda threads=16 blocks=16 fillblock_gpu!(p.cuJs[b], p.cupartitions[b], p.cumap, j_rowptr, j_colval, j_nzval, p.cupart, b)
        end
    end
    # Invert blocks" begin
    CUDA.@sync pivot, info = CUDA.CUBLAS.getrf_batched!(p.cuJs, true)
    CUDA.@sync pivot, info, p.cuJs = CUDA.CUBLAS.getri_batched(p.cuJs, pivot)
    p.P.nzVal .= 0.0
    # Move blocks to P" begin
    CUDA.@sync begin
        for b in 1:nblocks
            @cuda threads=16 blocks=16 fillP_gpu!(p.cuJs[b], p.cupartitions[b], p.cumap, p.P.rowPtr, p.P.colVal, p.P.nzVal, p.cupart, b)
        end
    end
    return p.P
end

"""
    function update(J::CuSparseMatrixCSR, p)

Update the preconditioner `p` from the sparse Jacobian `J` in CSR format for the GPU

1) The dense blocks `cuJs` are filled from the sparse Jacobian `J`
2) To a batch inversion of the dense blocks using CUBLAS
3) Extract the preconditioner matrix `p.P` from the dense blocks `cuJs`

"""
function update(J::CuSparseMatrixCSR, p)
    _update_gpu(J.rowPtr, J.colVal, J.nzVal, p)
end
function update(J::Transpose{T, CuSparseMatrixCSR{T}}, p) where T
    Jt = CuSparseMatrixCSC(J.parent)
    _update_gpu(Jt.colPtr, Jt.rowVal, Jt.nzVal, p)
end

function _update_cpu(colptr, rowval, nzval, p)
    nblocks = length(p.partitions)
    # Fill Block Jacobi
    @inbounds for b in 1:nblocks
        for i in p.partitions[b]
            for j in colptr[i]:colptr[i+1]-1
                if b == p.part[rowval[j]]
                    p.Js[b][p.map[rowval[j]], p.map[i]] = nzval[j]
                end
            end
        end
    end
    # Invert blocks
    for b in 1:nblocks
        p.Js[b] = inv(p.Js[b])
    end
    p.P.nzval .= 0.0
    # Move blocks to P
    @inbounds for b in 1:nblocks
        for i in p.partitions[b]
            for j in p.P.colptr[i]:p.P.colptr[i+1]-1
                if b == p.part[p.P.rowval[j]]
                    p.P.nzval[j] += p.Js[b][p.map[p.P.rowval[j]], p.map[i]]
                end
            end
        end
    end
    return p.P
end

"""
    function update(J::SparseMatrixCSC, p)

Update the preconditioner `p` from the sparse Jacobian `J` in CSC format for the CPU

Note that this implements the same algorithm as for the GPU and becomes very slow on CPU with growing number of blocks.

"""
function update(J::SparseMatrixCSC, p)
    _update_cpu(J.colptr, J.rowval, J.nzval, p)
end
function update(J::Transpose{T, SparseMatrixCSC{T, I}}, p) where {T, I}
    ix, jx, zx = findnz(J.parent)
    update(sparse(jx, ix, zx), p)
end

is_valid(precond::BlockJacobiPreconditioner) = _check_nan(precond.P)
_check_nan(P::SparseMatrixCSC) = !any(isnan.(P.nzval))
_check_nan(P::CuSparseMatrixCSR) = !any(isnan.(P.nzVal))

function Base.show(precond::BlockJacobiPreconditioner)
    npartitions = precond.npart
    nblock = div(size(precond.P, 1), npartitions)
    println("#partitions: $npartitions, Blocksize: n = ", nblock,
            " Mbytes = ", (nblock*nblock*npartitions*8.0)/(1024.0*1024.0))
    println("Block Jacobi block size: $(precond.nJs)")
end

