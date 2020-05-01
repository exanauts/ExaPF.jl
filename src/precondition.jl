module precondition

using LightGraphs
using Metis
using SparseArrays
using LinearAlgebra
using CuArrays
using CuArrays.CUSPARSE
export preconditioner
using CUDAnative

cuzeros = CuArrays.zeros

mutable struct partition
  npart::Int64
  partitions::Vector{Vector{Int64}}
  cupartitions::Vector{CuVector{Int64}}
  Js::Vector{Matrix{Float64}}
  cuJs::Vector{CuMatrix{Float64}}
  P
  function partition(J, blocks)
    adj = build_adjmatrix(J)
    g = Graph(adj)
    npart = blocks
    part = Metis.partition(g, npart)
    partitions = Vector{Vector{Int64}}()
    cupartitions = Vector{CuVector{Int64}}(undef, npart)
    for i in 1:npart
      push!(partitions, [])
    end
    for (i,v) in enumerate(part)
      push!(partitions[v], i)
    end
    for i in 1:npart
      cupartitions[i] = CuVector{Int64}(partitions[i])
    end
    Js = Vector{Matrix{Float64}}(undef, npart)
    for i in 1:npart
      Js[i] = zeros(Float64, length(partitions[i]), length(partitions[i]))
    end
    cuJs = Vector{CuMatrix{Float64}}(undef, npart)
    for i in 1:npart
      cuJs[i] = cuzeros(Float64, length(partitions[i]), length(partitions[i]))
    end
    P = CuSparseMatrixCSR(J)
    return new(npart, partitions, cupartitions, Js, cuJs, P)
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

  function create_preconditioner(J, p::partition)
    if J isa CuSparseMatrixCSR
      cscJ = collect(switch2csc(J))
    else
      cscJ = J
    end
    cscP = copy(cscJ)
    cscP.nzval .= 0 

    for i in 1:length(p.partitions)
      p.Js[i] = cscJ[p.partitions[i],p.partitions[i]]
      p.Js[i] = inv(p.Js[i])
    end
    for i in 1:length(p.partitions)
      cscP[p.partitions[i], p.partitions[i]] += p.Js[i]
    end
    if J isa CuSparseMatrixCSR
      p.P = CuSparseMatrixCSR(cscP)
    else
      p.P = cscP
    end
    return p.P
  end

  function mulinvP(y, x, p)
    y .= 0
    for i in 1:length(p.partitions)
      y[p.partitions[i]] = p.cuJs[i] * x[p.partitions[i]]
    end
    y
  end

  function mulinvP(x, p)
    res = similar(x)
    res .= 0.0
    for i in 1:length(p.partitions)
      res[p.partitions[i]] = p.cuJs[i] * x[p.partitions[i]]
    end
    res
  end

  function mulinvP!(y, x, p)
    y .= 0.0
    # index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # stride = blockDim().x * gridDim().x
    # y[p.partitions] .= p.cuJs * x[p.partitions]
      for i in 1:p.npart
        y[p.cupartitions[i]] .= p.cuJs[i] * x[p.cupartitions[i]]
        # @cuda threads=32 blocks=1 mulinvPpart!(y, x, p.cuJs[i], p.cupartitions[i])
      end
    return nothing
  end

  function mulinvP!(y, x, cuJs, cupartitions)
    y .= 0.0
    # index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # stride = blockDim().x * gridDim().x
    # y[p.partitions] .= p.cuJs * x[p.partitions]
      for i in 1:p.npart
        y[p.cupartitions[i]] .= p.cuJs[i] * x[p.cupartitions[i]]
        # @cuda threads=32 blocks=1 mulinvPpart!(y, x, p.cuJs[i], p.cupartitions[i])
      end
    return nothing
  end

  function mulinvPpart!(y, x, cuJ, partition)
    y[partition] = cuJ * x[partition]
    return nothing
  end
end