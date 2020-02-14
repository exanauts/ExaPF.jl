# Power flow module. The implementation is a modification of
# MATPOWER's code. We attach the original MATPOWER's license in makeYbus.m:
#
# MATPOWER
# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC)
# by Ray Zimmerman, PSERC Cornell
#
# Covered by the 3-clause BSD License.

module PowerFlow

using LinearAlgebra
using SparseArrays
using Printf
using Base

include("parse.jl")
include("parse_raw.jl")

"""
  assembleSbus(data)

Assembles vector of constant power injections (generator - load). Since
we do not have voltage-dependent loads, this vector only needs to be
assembled once at the beginning of the power flow routine.

"""
function assembleSbus(gen, load, SBASE, nbus)
  Sbus = zeros(Complex{Float64}, nbus)

  ngen = size(gen, 1)
  nload = size(load, 1)

  # retrieve indeces
  GEN_BUS, GEN_ID, GEN_PG, GEN_QG, GEN_QT, GEN_QB, GEN_STAT,
  GEN_PT, GEN_PB = idx_gen()

  LOAD_BUS, LOAD_ID, LOAD_STAT, LOAD_PL, LOAD_QL = idx_load()

  for i in 1:ngen
    if gen[i, GEN_STAT] == 1
      Sbus[gen[i, GEN_BUS]] += (gen[i, GEN_PG] + 1im*gen[i, GEN_QG])/SBASE
    end
  end

  for i in 1:nload
    if load[i, LOAD_STAT] == 1
      Sbus[load[i, LOAD_BUS]] -= (load[i, LOAD_PL] + 1im*load[i, LOAD_QL])/SBASE
    end
  end

  return Sbus
end


"""
  bustypeindex(data)

Returns vectors indexing buses by type: ref, pv, pq.

"""

function bustypeindex(bus, gen)

  # retrieve indeces
  BUS_B, BUS_AREA, BUS_VM, BUS_VA, BUS_NVHI, BUS_NVLO, BUS_EVHI,
  BUS_EVLO, BUS_TYPE = idx_bus()

  GEN_BUS, GEN_ID, GEN_PG, GEN_QG, GEN_QT, GEN_QB, GEN_STAT,
  GEN_PT, GEN_PB = idx_gen()
  
  # form vector that lists the number of generators per bus.
  # If a PV bus has 0 generators (e.g. due to contingency)
  # then that bus turns to a PQ bus.

  # Design note: this might be computed once and then modified for each contingency.

  gencon = zeros(Int8, size(bus, 1))

  for i in 1:size(gen, 1)
    if gen[i, GEN_STAT] == 1
      gencon[gen[i, GEN_BUS]] += 1
    end
  end

  bustype = copy(bus[:, BUS_TYPE])
  
  for i in 1:size(bus, 1)
    if (bustype[i] == 2) && (gencon[i] == 0)
      bustype[i] = 1
    elseif (bustype[i] == 1) && (gencon[i] > 0)
      bustype[i] = 2
    end
  end

  # form vectors
  ref = findall(x->x==3, bustype)
  pv = findall(x->x==2, bustype)
  pq = findall(x->x==1, bustype)



  return ref, pv, pq

end

"""
 residualFunction

Assembly residual function for N-R power flow
"""
function residualFunction(V, Ybus, Sbus, pv, pq)

  # form mismatch vector
  mis = V .* conj(Ybus * V) - Sbus

  # form residual vector
  F = [   real(mis[pv]);
  real(mis[pq]);
  imag(mis[pq]) ];

  return F
end


function residualFunction_real(v_re, v_im,
      ybus_re, ybus_im, pinj, qinj, pv, pq, nbus)

  npv = size(pv, 1)
  npq = size(pq, 1)

  F = zeros(npv + 2*npq)

  # REAL PV
  for i in 1:npv
    fr = pv[i]
    F[i] -= pinj[fr]
    for j in 1:nbus
      F[i] += (v_re[fr]*(v_re[j]*ybus_re[fr, j] - v_im[j]*ybus_im[fr, j]) +
               v_im[fr]*(v_im[j]*ybus_re[fr, j] + v_re[j]*ybus_im[fr, j]))
    end
  end

  # REAL PQ
  for i in 1:npq
    fr = pq[i]
    F[npv + i] -= pinj[fr]
    for j in 1:nbus
      F[npv + i] += (v_re[fr]*(v_re[j]*ybus_re[fr, j] - v_im[j]*ybus_im[fr, j]) +
               v_im[fr]*(v_im[j]*ybus_re[fr, j] + v_re[j]*ybus_im[fr, j]))
    end
  end
  
  # IMAG PQ
  for i in 1:npq
    fr = pq[i]
    F[npv + npq + i] -= qinj[fr]
    for j in 1:nbus
      F[npv + npq + i] += (v_im[fr]*(v_re[j]*ybus_re[fr, j] - v_im[j]*ybus_im[fr, j]) -
                     v_re[fr]*(v_im[j]*ybus_re[fr, j] + v_re[j]*ybus_im[fr, j]))
    end
  end

  return F
end


function residualJacobian(V, Ybus, pv, pq)
  n = size(V, 1)
  Ibus = Ybus*V
  diagV       = sparse(1:n, 1:n, V, n, n)
  diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
  diagVnorm   = sparse(1:n, 1:n, V./abs.(V), n, n)

  dSbus_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
  dSbus_dVa = 1im * diagV * conj(diagIbus - Ybus * diagV)

  j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
  j12 = real(dSbus_dVm[[pv; pq], pq])
  j21 = imag(dSbus_dVa[pq, [pv; pq]])
  j22 = imag(dSbus_dVm[pq, pq])

  J = [j11 j12; j21 j22]
end


function newtonpf(V, Ybus, data)

  # parameters NR
  tol = 1e-6
  maxiter = 10

  # iteration variables
  iter = 0
  converged = false

  # (TODO): Temporal hack
  ybus_re = Matrix(real(Ybus))
  ybus_im = Matrix(imag(Ybus))

  # data index
  BUS_B, BUS_AREA, BUS_VM, BUS_VA, BUS_NVHI, BUS_NVLO, BUS_EVHI,
  BUS_EVLO, BUS_TYPE = idx_bus()

  GEN_BUS, GEN_ID, GEN_PG, GEN_QG, GEN_QT, GEN_QB, GEN_STAT,
  GEN_PT, GEN_PB = idx_gen()

  LOAD_BUS, LOAD_ID, LOAD_STATUS, LOAD_PL, LOAD_QL = idx_load()

  bus = data["BUS"]
  gen = data["GENERATOR"]
  load = data["LOAD"]

  nbus = size(bus, 1)
  ngen = size(gen, 1)
  nload = size(load, 1)

  # retrieve ref, pv and pq index
  ref, pv, pq = bustypeindex(bus, gen)

  # retrieve power injections
  SBASE = data["CASE IDENTIFICATION"][1]
  Sbus = assembleSbus(gen, load, SBASE, nbus)
  pbus = real(Sbus)
  qbus = imag(Sbus)


  # voltage
  Vm = abs.(V)
  Va = angle.(V)

  # indices
  npv = size(pv, 1);
  npq = size(pq, 1);
  j1 = 1
  j2 = npv
  j3 = j2 + 1
  j4 = j2 + npq
  j5 = j4 + 1
  j6 = j4 + npq

  # form residual function
  F = residualFunction(V, Ybus, Sbus, pv, pq)

  # check for convergence
  normF = norm(F, Inf)
  @printf("Iteration %d. Residual norm: %g.\n", iter, normF)

  if normF < tol
    converged = true
  end

  while ((!converged) && (iter < maxiter))

    iter += 1

    # assemble jacobian and update
    J = residualJacobian(V, Ybus, pv, pq)
    dx = -(J \ F)

    # update voltage
    if (npv != 0)
      Va[pv] = Va[pv] + dx[j1:j2]
    end
    if (npq != 0)
      Va[pq] = Va[pq] + dx[j3:j4]
      Vm[pq] = Vm[pq] + dx[j5:j6]
    end

    for i in 1:size(V, 1)
      V[i] = Vm[i]*exp(1im*Va[i])
    end

    Vm = abs.(V)
    Va = angle.(V)

    # evaluate residual and check for convergence
    # F = residualFunction(V, Ybus, Sbus, pv, pq)
    
    v_re = real(V)
    v_im = imag(V)

    F = residualFunction_real(v_re, v_im,
            ybus_re, ybus_im, pbus, qbus, pv, pq, nbus)
    
    normF = norm(F, Inf)
    @printf("Iteration %d. Residual norm: %g.\n", iter, normF)

    if normF < tol
      converged = true
    end
  end

  if converged
    @printf("N-R converged in %d iterations.\n", iter)
  else
    @printf("N-R did not converge.\n")
  end

  return V, converged, normF

end

# end of module
end
