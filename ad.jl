module ad
using ForwardDiff
function myseed!(duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
               seeds::AbstractArray{ForwardDiff.Partials{N,V}}) where {T,V,N}

  @show size(duals)
  @show size(seeds)
  @show size(x)
  for i in 1:size(duals,1)
      duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
  end
  # @timeit timeroutput "myseed 2" begin
  #   zeroseed::ForwardDiff.Partials{N,V} = zero(ForwardDiff.Partials{N,V}) 
  #   for i in N+1:size(duals,1)
  #       duals[i] = ForwardDiff.Dual{T,V,N}(x[i], zeroseed)
  #   end
  # end
    return duals
end
end