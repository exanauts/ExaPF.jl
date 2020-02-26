module ad
using ForwardDiff
using CUDAnative, CuArrays
function myseed!(duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
               seeds::AbstractArray{ForwardDiff.Partials{N,V}}) where {T,V,N}

  for i in 1:size(duals,1)
#   for i in 1:size(duals,1)
      duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
  end
  return nothing
end
function cumyseed!(duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
               seeds::AbstractArray{ForwardDiff.Partials{N,V}}) where {T,V,N}

  index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  stride = blockDim().x * gridDim().x
  for i in index:stride:size(duals,1)
#   for i in 1:size(duals,1)
      duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
      # duals[i].value = x[i]
  end
  return nothing
end
end