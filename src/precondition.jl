module precondition

using LightGraphs
using Metis
using SparseArrays
using LinearAlgebra
using CuArrays
using CuArrays.CUSPARSE
export preconditioner

mutable struct partition
  npart::Int64
  partitions::Vector{Vector{Int64}}
  Js::Vector{Matrix{Float64}}
  P
  function partition(J, blocks)
    adj = build_adjmatrix(J)
    g = Graph(adj)
    npart = blocks
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
    P = copy(J)
    P.nzVal .= 0 
    return new(npart, partitions, Js, P)
  end
end

  function build_adjmatrix(A)
      rows = Int64[]
      cols = Int64[]
      vals = Float64[]
      colsA = A.colVal
      m, n = size(A)
      for i = 1:n
          for j in 1:length(A.nzVal)
              push!(rows, colsA[j])
              push!(cols, i)
              push!(vals, 1.0)

              push!(rows, i)
              push!(cols, colsA[j])
              push!(vals, 1.0)
          end
      end
      return sparse(rows,cols,vals,size(A,1),size(A,2))
  end

  function create_preconditioner(J, p::partition)
    cscJ = collect(switch2csc(J))
    cscP = copy(cscJ)
    cscP.nzval .= 0 

    for i in 1:length(p.partitions)
      p.Js[i] = cscJ[p.partitions[i],p.partitions[i]]
      p.Js[i] = inv(p.Js[i])
    end
    for i in 1:length(p.partitions)
      cscP[p.partitions[i], p.partitions[i]] += p.Js[i]
    end
    p.P = CuSparseMatrixCSR(cscP)
    return p.P
  end
end