module kernels
  using CUDAnative, CuArrays
  export sync, dispatch, generate, togenerate

  togenerate = "cuda"

  macro sync(type, expr)
    ex = quote 
      if $type == CuArray || $type == :CuArray || $type == CuArrays.CuArray
        CuArrays.@sync begin
        $expr
        end
      end
      if $type == Array || $type == :Array 
        $expr
      end
    end
    return esc(ex)
  end

  macro dispatch(type, threads, blocks, expr)
    cuda = Meta.parse("kernels.$expr")
    cpu = Meta.parse("kernels.$expr")
    ex = nothing
    ex = quote 
      if $type == CuArray || $type == :CuArray || $type == CuArrays.CuArray
          @cuda $threads $blocks $cuda
      end
      if $type == Array || $type == :Array
          $cpu
      end
    end
    return esc(ex)
  end

  macro getstrideindex()
    @show togenerate
    ex = nothing
    if togenerate == "cuda"
      ex = quote 
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
      end
    end
    if togenerate == "cpu"
      ex = quote
        index  = 1
        stride = 1
      end
    end
    return esc(ex)
  end

  function residualFunction_polar!(F, v_m, v_a,
                                  ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                                  ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                                  pinj, qinj, pv, pq, nbus)

  npv = size(pv, 1)
  npq = size(pq, 1)

  @getstrideindex()

  # REAL PV
  for i in index:stride:npv
      fr = pv[i]
      F[i] -= pinj[fr]
      for (j,c) in enumerate(ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1)
      to = ybus_re_rowval[c]
      aij = v_a[fr] - v_a[to]
      F[i] += v_m[fr]*v_m[to]*(ybus_re_nzval[c]*CUDAnative.cos(aij) + ybus_im_nzval[c]*CUDAnative.sin(aij))
      # F[i] += v_m[fr]*v_m[to]*(ybus_re_nzval[c]*cos(aij) + ybus_im_nzval[c]*sin(aij))
      end
  end

  # REAL PQ
  for i in index:stride:npq
      fr = pq[i]
      F[npv + i] -= pinj[fr]
      for (j,c) in enumerate(ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1)
      to = ybus_re_rowval[c]
      aij = v_a[fr] - v_a[to]
      F[npv + i] += v_m[fr]*v_m[to]*(ybus_re_nzval[c]*CUDAnative.cos(aij) + ybus_im_nzval[c]*CUDAnative.sin(aij))
      # F[npv + i] += v_m[fr]*v_m[to]*(ybus_re_nzval[c]*cos(aij) + ybus_im_nzval[c]*sin(aij))
      end
  end

  # IMAG PQ
  for i in index:stride:npq
      fr = pq[i]
      F[npv + npq + i] -= qinj[fr]
      for (j,c) in enumerate(ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1)
      to = ybus_re_rowval[c]
      aij = v_a[fr] - v_a[to]
      F[npv + npq + i] += v_m[fr]*v_m[to]*(ybus_re_nzval[c]*CUDAnative.sin(aij) - ybus_im_nzval[c]*CUDAnative.cos(aij))
      # F[npv + npq + i] += v_m[fr]*v_m[to]*(ybus_re_nzval[c]*sin(aij) - ybus_im_nzval[c]*cos(aij))
      end
  end

  return nothing
  end

  using ForwardDiff
  
  function myseed!(duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
                seeds::AbstractArray{ForwardDiff.Partials{N,V}}) where {T,V,N}

    @getstrideindex()

    for i in index:stride:size(duals,1)
  #   for i in 1:size(duals,1)
        duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
        # duals[i].value = x[i]
    end
    return nothing
  end

  function getpartials(compressedJ, t1sF)

    @getstrideindex()

    for i in index:stride:size(t1sF,1) # Go over outputs
      compressedJ[:,i] .= ForwardDiff.partials.(t1sF[i]).values
    end
  end

  function uncompress(J_nzVal, J_rowPtr, J_colVal, compressedJ, coloring, nmap)

    @getstrideindex()

    for i in index:stride:nmap
      for j in J_rowPtr[i]:J_rowPtr[i+1]-1
        J_nzVal[j] = compressedJ[coloring[J_colVal[j]], i]
      end
    end
  end
end