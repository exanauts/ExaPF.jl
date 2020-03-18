module iterative

include("precondition.jl")
using LinearAlgebra
using .precondition

export bicgstab

mulinvP = precondition.mulinvP

"""
  bicgstab according to 

  Van der Vorst, Henk A. 
  "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems." 
  SIAM Journal on scientific and Statistical Computing 13, no. 2 (1992): 631-644.
"""

function bicgstab(A, b, p ; tol = 1e-6, maxiter = size(A,1))
  n    = size(b, 1)
  x0   = rand(Float64, n)
  r0   = b - A * x0
  br0  = r0
  rho0 = alpha = omega0 = 1
  v0   = p0 = zeros(Float64, n)

  ri     = r0
  rhoi   = rho0
  omegai = omega0
  vi = v0
  pi = p0
  xi = x0

  go = true
  iter = 1
  while go
    rhoi1 = dot(br0, ri) ; beta = (rhoi1/rhoi) * (alpha / omegai)
    pi1 = ri + beta * (pi - omegai .* vi)
    y = mulinvP(pi1, p)
    vi1 = A * y
    alpha = rhoi1 / dot(br0, vi1)
    s = ri - alpha * vi1
    z = mulinvP(s, p)
    t = A * z
    omegai1 = dot(mulinvP(t, p), mulinvP(s, p)) / dot(mulinvP(t, p), mulinvP(t, p))
    xi1 = xi + alpha * y + omegai1 * z
    if norm((A * xi1) - b) < tol
      go = false
      println("Tolerance reached at iteration $iter")
    end
    if maxiter == iter
      @show iter
      go = false
      println("Not converged")
    end
    ri     = s - omegai1 * t

    rhoi   = rhoi1
    pi     = pi1
    vi     = vi1
    omegai = omegai1
    xi     = xi1
    iter   += 1
  end
  return xi
end

end
