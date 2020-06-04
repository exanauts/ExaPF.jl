module Iterative

include("algorithms/precondition.jl")
using LinearAlgebra
using .Precondition
using CuArrays
using CUDAnative
using TimerOutputs
using SparseArrays

cuzeros = CuArrays.zeros


export bicgstab

# mulinvP = Precondition.mulinvP
# mulinvP! = Precondition.mulinvP!

"""
  bicgstab according to 

  Van der Vorst, Henk A. 
  "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems." 
  SIAM Journal on scientific and Statistical Computing 13, no. 2 (1992): 631-644.
"""

function bicgstab(A, b, P, to = nothing; tol = 1e-6, maxiter = size(A,1),
                 verbose=false)
  
  # parameters
  n    = size(b, 1)
  x0   = similar(b)
  x0 = P * b
  # mulinvP!(x0, b, p)
  r0   = b - A * x0
  br0  = copy(r0)
  rho0 = 1.0
  alpha = 1.0
  omega0 = 1.0
  if A isa SparseArrays.SparseMatrixCSC
    v0   = p0 = zeros(Float64, n)
  else
    v0   = p0 = cuzeros(Float64, n)
  end


  ri     = copy(r0)
  rhoi   = copy(rho0)
  omegai = copy(omega0)
  vi = copy(v0)
  pi = copy(p0)
  xi = copy(x0)

  y = similar(pi)
  s = similar(pi)
  z = similar(pi)
  t1 = similar(pi)
  t2 = similar(pi)
  pi1 = similar(pi)
  vi1 = similar(pi)
  xi1 = similar(pi)
  t = similar(pi)

  go = true
  iter = 1
  while go

    rhoi1 = dot(br0, ri) ; beta = (rhoi1/rhoi) * (alpha / omegai)
    pi1 .= ri .+ beta .* (pi .- omegai .* vi)
    y .= P * pi1
    vi1 .= A * y
    alpha = rhoi1 / dot(br0, vi1)
    s .= ri .- (alpha * vi1)
    z .= P * s
    t .= A * z
    t1 .= P * t
    t2 .= P * s
    omegai1 = dot(t1, t2) / dot(t1, t1)
    xi1 .= xi .+ alpha .* y .+ omegai1 .* z
  
    anorm = norm((A * xi1) .- b)

    if verbose
      println("\tIteration: ", iter)
      println("\tAbsolute norm: ", anorm)
    end

    if anorm < tol
      go = false
      println("Tolerance reached at iteration $iter")
    end

    if maxiter == iter
      @show iter
      go = false
      println("Not converged")
    end
    
    ri     .= s .- omegai1 .* t
    rhoi   = rhoi1
    pi     .= pi1
    vi     .= vi1
    omegai = omegai1
    xi     .= xi1
    iter   += 1
  end

  return xi, iter

end

end
