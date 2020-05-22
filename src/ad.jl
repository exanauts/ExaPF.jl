module AD
include("target/kernels.jl")
using ForwardDiff
using CUDAnative, CuArrays
using .Kernels
using TimerOutputs
using SparseArrays
  
function myseed!(duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
              seeds::AbstractArray{ForwardDiff.Partials{N,V}}) where {T,V,N}

  Kernels.@getstrideindex()

  for i in index:stride:size(duals,1)
#   for i in 1:size(duals,1)
      duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
      # duals[i].value = x[i]
  end
  return nothing
end

function getpartials(compressedJ, t1sF)

  Kernels.@getstrideindex()

  for i in index:stride:size(t1sF,1) # Go over outputs
    compressedJ[:,i] .= ForwardDiff.partials.(t1sF[i]).values
  end
end

function uncompress(J_nzVal, J_rowPtr, J_colVal, compressedJ, coloring, nmap)

  Kernels.@getstrideindex()

  for i in index:stride:nmap
    for j in J_rowPtr[i]:J_rowPtr[i+1]-1
      J_nzVal[j] = compressedJ[coloring[J_colVal[j]], i]
    end
  end
end

function residualJacobianAD!(J, residualFunction_polar!, arrays, coloring, v_m, v_a,
                            ybus_re, ybus_im, pinj, qinj, pv, pq, nbus, to = nothing)
  @timeit to "Before" begin
  T = typeof(arrays.x).name.name
  nv_m = size(v_m, 1)
  nv_a = size(v_a, 1)
  nmap = size(arrays.map, 1)
  nthreads=256
  nblocks=ceil(Int64, nmap/nthreads)
  n = nv_m + nv_a
  ncolor = size(unique(coloring),1)
  arrays.x[1:nv_m] .= v_m
  arrays.t1sx .= arrays.x
  arrays.x[nv_m+1:nv_m+nv_a] .= v_a
  arrays.t1sF .= 0.0
  @timeit to "Seeding" begin
  Kernels.@sync begin
    Kernels.@dispatch threads=nthreads blocks=nblocks myseed!(arrays.t1svarx, arrays.varx, arrays.t1sseeds)
  end
  end
  nthreads=256
  nblocks=ceil(Int64, nbus/nthreads)
  end

  @timeit to "Function" begin
  Kernels.@sync begin
    Kernels.@dispatch threads=nthreads blocks=nblocks residualFunction_polar!(arrays.t1sF, arrays.t1sx[1:nv_m], arrays.t1sx[nv_m+1:nv_m+nv_a],
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval, 
        ybus_im.nzval, ybus_im.colptr, ybus_im.rowval, 
        pinj, qinj, pv, pq, nbus)
  end
  end

  @timeit to "Get partials" begin
  Kernels.@sync begin
    Kernels.@dispatch threads=nthreads blocks=nblocks getpartials(arrays.compressedJ, arrays.t1sF)
  end
  end
  @timeit to "Uncompress" begin
  # Uncompress matrix. Sparse matrix elements have different names with CUDA
  if J isa SparseArrays.SparseMatrixCSC
    for i in 1:nmap
      for j in J.colptr[i]:J.colptr[i+1]-1
        J.nzval[j] = arrays.compressedJ[coloring[i],J.rowval[j]]
      end
    end
  end
  if J isa CuArrays.CUSPARSE.CuSparseMatrixCSR
    Kernels.@sync begin
      Kernels.@dispatch threads=nthreads blocks=nblocks uncompress(
            J.nzVal, J.rowPtr, J.colVal, arrays.compressedJ, coloring, nmap)
    end
  end
  # @show J.nzVal
  return J
end
end

struct comparrays
  t1sseeds
  compressedJ
  t1sF
  x
  t1sx
  varx
  t1svarx
  map
end

function createArrays(coloring, F, v_m, v_a, pv, pq)
  nv_m = size(v_m, 1)
  nv_a = size(v_a, 1)
  n = nv_m + nv_a
  ncolor = size(unique(coloring),1)
  if F isa Array
    T = Vector
    M = Matrix
    A = Array
  elseif F isa CuArray
    T = CuVector
    M = CuMatrix
    A = CuArray
  else
    error("Wrong array type ", typeof(F))
  end

  mappv = [i + nv_m for i in pv]
  mappq = [i + nv_m for i in pq]
  map = T{Int64}(vcat(mappv, mappq, pq))
  nmap = size(map,1)

  t1s{N} =  ForwardDiff.Dual{Nothing,Float64, N} where N
  x = T{Float64}(undef, nv_m + nv_a)
  t1sx = T{t1s{ncolor}}(x)
  t1sF = T{t1s{ncolor}}(undef, nmap)
  varx = view(x,map)
  t1sseedvec = zeros(Float64, ncolor)
  t1sseeds = T{ForwardDiff.Partials{ncolor,Float64}}(undef, nmap)
  for i in 1:nmap
    for j in 1:ncolor
      if coloring[i] == j
        t1sseedvec[j] = 1.0
      end
    end
    t1sseeds[i] = ForwardDiff.Partials{ncolor, Float64}(NTuple{ncolor, Float64}(t1sseedvec))
    t1sseedvec .= 0
  end
  compressedJ = M{Float64}(undef, ncolor, nmap)
  t1svarx = view(t1sx, map)
  nthreads=256
  nblocks=ceil(Int64, nmap/nthreads)
  # CuArrays.@sync begin
  # ad.myseed!(t1svarx, varx, t1sseeds)
  # end
  return comparrays(t1sseeds, compressedJ, t1sF, x, t1sx, varx, t1svarx, map)
end
end