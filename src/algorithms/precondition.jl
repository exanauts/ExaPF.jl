module Precondition

include("../target/kernels.jl")
using LightGraphs
using Metis
using SparseArrays
using LinearAlgebra
using CUDA
using CUDA.CUSPARSE
using TimerOutputs
using .Kernels

cuzeros = CUDA.zeros

abstract type AbstractPreconditioner end

struct NoPreconditioner <: AbstractPreconditioner end

mutable struct Preconditioner <: AbstractPreconditioner
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
    function Preconditioner(J, npart)
        adj = build_adjmatrix(J)
        g = Graph(adj)
        m = size(J,1)
        n = size(J,2)
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
        println("Block Jacobi block size: $nJs")
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
        # TODO: clean global variables
        if Main.target == "cuda"
            global cupartitions = Vector{CuVector{Int64}}(undef, npart)
            for i in 1:npart
                cupartitions[i] = CuVector{Int64}(partitions[i])
            end
            global cuJs = Vector{CuMatrix{Float64}}(undef, length(partitions))
            for i in 1:length(partitions)
                cuJs[i] = CuMatrix{Float64}(I, nJs, nJs)
            end
            global cumap = cu(map)
            global cupart = cu(part)
            global P = CuSparseMatrixCSR(P)
        else
            global cuJs = nothing
            global cupartitions = nothing
            global cumap = nothing
            global cupart = nothing
        end
        return new(npart, nJs, partitions, cupartitions, Js, cuJs, map, cumap, part, cupart, P)
    end
end

function build_adjmatrix(A)
    rows = Int64[]
    cols = Int64[]
    vals = Float64[]
    @show size(A)
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
    println("Creating matrix")
    return sparse(rows,cols,vals,size(A,1),size(A,2))
end

function fillblock_gpu!(cuJs, partition, map, rowPtr, colVal, nzVal, part, b)

    Kernels.@getstrideindex()

    for i in index:stride:length(partition)
        for j in rowPtr[partition[i]]:rowPtr[partition[i]+1]-1
            if b == part[colVal[j]]
                @inbounds cuJs[map[partition[i]], map[colVal[j]]] = nzVal[j]
            end
        end
    end
    return nothing
end

function fillP_gpu!(cuJs, partition, map, rowPtr, colVal, nzVal, part, b)

    Kernels.@getstrideindex()

    for i in index:stride:length(partition)
        for j in rowPtr[partition[i]]:rowPtr[partition[i]+1]-1
            if b == part[colVal[j]]
                @inbounds nzVal[j] += cuJs[map[partition[i]], map[colVal[j]]]
            end
        end
    end
    return nothing
end

function update(J, p, to)
    m = size(J, 1)
    n = size(J, 2)
    nblocks = length(p.partitions)
    if J isa CuSparseMatrixCSR
        @timeit to "Fill Block Jacobi" begin
            Kernels.@sync begin
                for b in 1:nblocks
                    Kernels.@dispatch threads=16 blocks=16 fillblock_gpu!(p.cuJs[b], p.cupartitions[b], p.cumap, J.rowPtr, J.colVal, J.nzVal, p.cupart, b)
                end
            end
        end
        @timeit to "Invert blocks" begin
            CUDA.@sync pivot, info = CUDA.CUBLAS.getrf_batched!(p.cuJs, true)
            CUDA.@sync pivot, info, p.cuJs = CUDA.CUBLAS.getri_batched(p.cuJs, pivot)
        end
        p.P.nzVal .= 0.0
        @timeit to "Move blocks to P" begin
            Kernels.@sync begin
                for b in 1:nblocks
                    Kernels.@dispatch threads=16 blocks=16 fillP_gpu!(p.cuJs[b], p.cupartitions[b], p.cumap, p.P.rowPtr, p.P.colVal, p.P.nzVal, p.cupart, b)
                end
            end
        end
    else
        @timeit to "Fill Block Jacobi" begin
            for b in 1:nblocks
                for i in p.partitions[b]
                    for j in J.colptr[i]:J.colptr[i+1]-1
                        if b == p.part[J.rowval[j]]
                            p.Js[b][p.map[J.rowval[j]], p.map[i]] = J.nzval[j]
                        end
                    end
                end
            end
        end
        @timeit to "Invert blocks" begin
            for b in 1:nblocks
                p.Js[b] = inv(p.Js[b])
            end
        end
        p.P.nzval .= 0.0
        @timeit to "Move blocks to P" begin
            for b in 1:nblocks
                for i in p.partitions[b]
                    for j in p.P.colptr[i]:p.P.colptr[i+1]-1
                        if b == p.part[p.P.rowval[j]]
                            p.P.nzval[j] += p.Js[b][p.map[p.P.rowval[j]], p.map[i]]
                        end
                    end
                end
            end
        end
    end
    return p.P
end
end
