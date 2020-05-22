using LineSearches
using LinearAlgebra


function ls(uk::Vector{Float64}, grad_L::Vector{Float64}, Lu::Function, grad_Lu::Function)
  s = copy(-grad_L)
  Lalpha(alpha) = Lu(uk .+ alpha.*s)
  function grad_Lalpha(alpha)
    return dot(grad_Lu(uk .+ alpha .* s), s)
  end
  function Lgrad_Lalpha(alpha)
    gvec = grad_Lu(uk .+ alpha .* s)
    phi = Lu(uk .+ alpha .*s)
    dphi = dot(gvec, s)
    return (phi, dphi)
  end
  dL_0 = dot(s, grad_L)
  obj = Lu(uk)
  ls_ = BackTracking()
  alpha, obj = ls_(Lalpha, grad_Lalpha, Lgrad_Lalpha, 1.0, obj, dL_0)
  return alpha
end
