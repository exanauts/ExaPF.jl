module preconditioner

using LightGraphs
using Metis
using SparseArrays
using LinearAlgebra

export preconditioner

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

  function precondition(J, blocks)
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

    for i in 1:npart
      Js[i] = J[partitions[i],partitions[i]]
      Js[i] = inv(Js[i])
    end
    P = similar(J)
    P .= 0 
    for i in 1:npart
      P[partitions[i], partitions[i]] += Js[i]
    end
    return P
  end
end