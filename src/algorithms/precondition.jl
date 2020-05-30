module Precondition

include("../target/kernels.jl")
using LightGraphs
using Metis
using SparseArrays
using LinearAlgebra
using CuArrays
using CuArrays.CUSPARSE
using CUDAnative
using .Kernels
# using CUDA

cuzeros = CuArrays.zeros

mutable struct Preconditioner
  npart::Int64
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
    for i in 1:npart
      Js[i] = zeros(Float64, length(partitions[i]), length(partitions[i]))
    end
    if Main.target == "cuda"
      global cupartitions = Vector{CuVector{Int64}}(undef, npart)
      for i in 1:npart
        cupartitions[i] = CuVector{Int64}(partitions[i])
      end
      global cuJs = Vector{CuMatrix{Float64}}(undef, length(partitions))
      for i in 1:length(partitions)
        cuJs[i] = cuzeros(Float64, length(partitions[i]), length(partitions[i]))
        # cuJs[i][:] = J[partitions[i],partitions[i]]
      end

      rowPtr = CuVector{Cint}(undef, npart+1)
      for i in 1:npart+1 rowPtr[i] = Cint(i) end
      colVal = CuVector{Cint}(undef, npart)
      for i in 1:npart colVal[i] = Cint(i) end

      blockDim::Cint = ceil(m/npart) 
      nzVal = CuVector{Float64}(undef, npart*blockDim^2)
      dims::NTuple{2,Int} = (m,n)
      dir = 'R'
      nnz::Cint = blockDim^2*npart
      P = CuSparseMatrixBSR{Float64}(rowPtr, colVal, nzVal, dims::NTuple{2,Int},blockDim::Cint, dir, nnz::Cint)
      global cumap = cu(map)
      global cupart = cu(part)
    else
      global cuJs = nothing
      global cupartitions = nothing
      global cumap = nothing
      global cupart = nothing
      global P = copy(J)
    end
    return new(npart, partitions, cupartitions, Js, cuJs, map, cumap, part, cupart, P)# cuJs, P)
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

  function update(jacobianAD, p::Preconditioner)
    m = size(jacobianAD.J,1)
    n = size(jacobianAD.J,2)
    nblocks = length(p.partitions)
    if jacobianAD.J isa CuSparseMatrixCSR
      for i in 1:nblocks
        p.cuJs[i][:] = jacobianAD.J[p.cupartitions[i],p.cupartitions[i]]
      end
      # p.cuJs[:] .= J[p.partitions,p.partitions]
      pivot, info = CuArrays.CUBLAS.getrf_batched!(p.cuJs, true)
      CuArrays.@sync pivot, info, cuJs = CuArrays.CUBLAS.getri_batched(p.cuJs, pivot)
      sqblockdim = Int(m/nblocks)^2 
      for i in 1:nblocks
        p.P.nzVal[(i-1)*sqblockdim+1:i*sqblockdim] = p.cuJs[i][:]
      end
    else
      partitions = p.partitions
      coloring = jacobianAD.coloring
      J = jacobianAD.J 
      nmap = 0
      for b in partitions
        nmap += length(b)
      end
      map = Vector{Int64}(undef, nmap)
      part = Vector{Int64}(undef, nmap)
      for b in 1:nblocks
        for (i,el) in enumerate(partitions[b])
          map[el] = i
          part[el] = b
        end
      end
      for i in 1:m
        for j in J.colptr[i]:J.colptr[i+1]-1
          if part[i] == part[J.rowval[j]]
            p.Js[part[i]][map[jacobianAD.J.rowval[j]], map[i]] = jacobianAD.J.nzval[j]
          end
        end
      end
      # for b in 1:nblocks
      #   p.Js[b] = jacobianAD.J[p.partitions[b],p.partitions[b]]
      # end
      for b in 1:nblocks
        p.Js[b] = inv(p.Js[b])
      end
      p.P.nzval .= 0 
      for i in 1:nblocks
        p.P[p.partitions[i], p.partitions[i]] += p.Js[i]
      end
    end
    return p.P
  end
  # function uncompress(J_nzVal, J_rowPtr, J_colVal, compressedJ, coloring, nmap)

  #   Kernels.@getstrideindex()

  #   for i in index:stride:nmap
  #     for j in J_rowPtr[i]:J_rowPtr[i+1]-1
  #       J_nzVal[j] = compressedJ[coloring[J_colVal[j]], i]
  #     end
  #   end
  # end
end
