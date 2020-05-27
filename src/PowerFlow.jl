# Power flow module. The implementation is a modification of
# MATPOWER's code. We attach the original MATPOWER's license in makeYbus.m:
#
# MATPOWER
# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC)
# by Ray Zimmerman, PSERC Cornell
#
# Covered by the 3-clause BSD License.
__precompile__(false)
module PowerFlow

export Pf

include("parse/parse.jl")
include("parse/parse_raw.jl")
include("ad.jl")
include("target/kernels.jl")
include("algorithms/precondition.jl")
include("iterative.jl")
include("network.jl")
using ForwardDiff
using LinearAlgebra
using SparseArrays
using Printf
using CuArrays
using CuArrays.CUSPARSE
using CuArrays.CUSOLVER
using TimerOutputs
using CUDAnative
to = TimerOutput()
using SparseDiffTools
using IterativeSolvers
using .AD
using .Kernels
using .Precondition
using .Iterative
using Krylov

struct Pf
  V::Array{Complex{Float64}}
  Ybus::SparseArrays.SparseMatrixCSC{Complex{Float64},Int64}
  data::Dict{String,Array}
end

mutable struct Spmat{T}
  colptr
  rowval
  nzval

  # function spmat{T}(colptr::Vector{Int64}, rowval::Vector{Int64}, nzval::Vector{T}) where T
  function Spmat{T}(mat::SparseMatrixCSC{Complex{Float64}, Int}) where T
    matreal = new(T{Int64}(mat.colptr), T{Int64}(mat.rowval), T{Float64}(real.(mat.nzval)))
    matimag = new(T{Int64}(mat.colptr), T{Int64}(mat.rowval), T{Float64}(imag.(mat.nzval)))
    return matreal, matimag
  end
end
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
  GEN_PT, GEN_PB = Parse.idx_gen()

  LOAD_BUS, LOAD_ID, LOAD_STAT, LOAD_PL, LOAD_QL = Parse.idx_load()

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
  BUS_EVLO, BUS_TYPE = Parse.idx_bus()

  GEN_BUS, GEN_ID, GEN_PG, GEN_QG, GEN_QT, GEN_QB, GEN_STAT,
  GEN_PT, GEN_PB = Parse.idx_gen()
  
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


function residualFunction_real!(F, v_re, v_im,
      ybus_re, ybus_im, pinj, qinj, pv, pq, nbus)

  npv = size(pv, 1)
  npq = size(pq, 1)

  # REAL PV
  for i in 1:npv
    fr = pv[i]
    F[i] -= pinj[fr]
    for (j,c) in enumerate(ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1)
      to = ybus_re.rowval[c]
      F[i] += (v_re[fr]*(v_re[to]*ybus_re.nzval[c] - v_im[to]*ybus_im.nzval[c]) +
               v_im[fr]*(v_im[to]*ybus_re.nzval[c] + v_re[to]*ybus_im.nzval[c]))
    end
  end

  # REAL PQ
  for i in 1:npq
    fr = pq[i]
    F[npv + i] -= pinj[fr]
    for (j,c) in enumerate(ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1)
      to = ybus_re.rowval[c]
      F[npv + i] += (v_re[fr]*(v_re[to]*ybus_re.nzval[c] - v_im[to]*ybus_im.nzval[c]) +
                     v_im[fr]*(v_im[to]*ybus_re.nzval[c] + v_re[to]*ybus_im.nzval[c]))
    end
  end
  
  # IMAG PQ
  for i in 1:npq
    fr = pq[i]
    F[npv + npq + i] -= qinj[fr]
    for (j,c) in enumerate(ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1)
      to = ybus_re.rowval[c]
      F[npv + npq + i] += (v_im[fr]*(v_re[to]*ybus_re.nzval[c] - v_im[to]*ybus_im.nzval[c]) -
                           v_re[fr]*(v_im[to]*ybus_re.nzval[c] + v_re[to]*ybus_im.nzval[c]))
    end
  end

  return F
end

function residualFunction_polar!(F, v_m, v_a,
                                ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                                ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                                pinj, qinj, pv, pq, nbus)

npv = size(pv, 1)
npq = size(pq, 1)

Kernels.@getstrideindex()

# REAL PV
for i in index:stride:npv
  fr = pv[i]
  F[i] -= pinj[fr]
  for (j,c) in enumerate(ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1)
  to = ybus_re_rowval[c]
  aij = v_a[fr] - v_a[to]
  F[i] += v_m[fr]*v_m[to]*(ybus_re_nzval[c]*Kernels.@cos(aij) + ybus_im_nzval[c]*Kernels.@sin(aij))
  end
end

# REAL PQ
for i in index:stride:npq
  fr = pq[i]
  F[npv + i] -= pinj[fr]
  for (j,c) in enumerate(ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1)
  to = ybus_re_rowval[c]
  aij = v_a[fr] - v_a[to]
  F[npv + i] += v_m[fr]*v_m[to]*(ybus_re_nzval[c]*Kernels.@cos(aij) + ybus_im_nzval[c]*Kernels.@sin(aij))
  end
end

# IMAG PQ
for i in index:stride:npq
  fr = pq[i]
  F[npv + npq + i] -= qinj[fr]
  for (j,c) in enumerate(ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1)
  to = ybus_re_rowval[c]
  aij = v_a[fr] - v_a[to]
  F[npv + npq + i] += v_m[fr]*v_m[to]*(ybus_re_nzval[c]*Kernels.@sin(aij) - ybus_im_nzval[c]*Kernels.@cos(aij))
  end
end

return nothing
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


function solve(pf::Pf, npartitions = 2, solver="gmres")
  # Set array type
  # For CPU choose Vector and SparseMatrixCSC
  # For GPU choose CuVector and SparseMatrixCSR (CSR!!! Not CSC)
  println("Target set to $(Main.target)")
  if Main.target == "cpu"
    T = Vector
    M = SparseMatrixCSC
    A = Array
  end
  if Main.target == "cuda"
    T = CuVector
    M = CuSparseMatrixCSR
    A = CuArray
  end

  V = pf.V
  data = pf.data
  Ybus = pf.Ybus

  V = T(V)

  # parameters NR
  tol = 1e-6
  maxiter = 20

  # iteration variables
  iter = 0
  converged = false

  ybus_re, ybus_im = Spmat{T}(Ybus)

  # data index
  BUS_B, BUS_AREA, BUS_VM, BUS_VA, BUS_NVHI, BUS_NVLO, BUS_EVHI,
  BUS_EVLO, BUS_TYPE = Parse.idx_bus()

  GEN_BUS, GEN_ID, GEN_PG, GEN_QG, GEN_QT, GEN_QB, GEN_STAT,
  GEN_PT, GEN_PB = Parse.idx_gen()

  LOAD_BUS, LOAD_ID, LOAD_STATUS, LOAD_PL, LOAD_QL = Parse.idx_load()

  bus = data["BUS"]
  gen = data["GENERATOR"]
  load = data["LOAD"]

  nbus = size(bus, 1)
  ngen = size(gen, 1)
  nload = size(load, 1)

  # retrieve ref, pv and pq index
  ref, pv, pq = bustypeindex(bus, gen)
  pv = T(pv)
  pq = T(pq)

  # retrieve power injections
  SBASE = data["CASE IDENTIFICATION"][1]
  Sbus = assembleSbus(gen, load, SBASE, nbus)
  pbus = T(real(Sbus))
  qbus = T(imag(Sbus))

  # voltage
  Vm = abs.(V)
  Va = Kernels.@angle(V)

  # Number of GPU threads
  nthreads=256
  nblocks=ceil(Int64, nbus/nthreads)

  # indices
  npv = size(pv, 1);
  npq = size(pq, 1);
  j1 = 1
  j2 = npv
  j3 = j2 + 1
  j4 = j2 + npq
  j5 = j4 + 1
  j6 = j4 + npq

  # v_re[:] = real(V)
  # v_im[:] = imag(V)

  # form residual function
  F = T(zeros(Float64, npv + 2*npq))
  dx = similar(F)
  Kernels.@sync begin
  Kernels.@dispatch threads=nthreads blocks=nblocks residualFunction_polar!(F, Vm, Va,
                          ybus_re.nzval, ybus_re.colptr, ybus_re.rowval, 
                          ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
                          pbus, qbus, pv, pq, nbus)
  end

  J = residualJacobian(V, Ybus, pv, pq)
  dim_J = size(J, 1)
  @show npartitions
  nblock = size(J,1)/npartitions
  println("Blocksize: n = ", nblock, " Mbytes = ", (nblock*nblock*npartitions*8.0)/1024.0/1024.0)
  println("Partitioning...")
  preconditioner = Precondition.Preconditioner(J, npartitions)
  println("$npartitions partitions created")
  println("Coloring...")
  @timeit to "Coloring" coloring = T{Int64}(matrix_colors(J))
  ncolors = size(unique(coloring),1)
  println("Number of Jacobian colors: ", ncolors)
  J = M(J)
  println("Creating JacobianAD...")
  jacobianAD = AD.JacobianAD(J, coloring, F, Vm, Va, pv, pq)

  # check for convergence
  normF = norm(F, Inf)
  @printf("Iteration %d. Residual norm: %g.\n", iter, normF)

  if normF < tol
    converged = true
  end

  linsol_iters = []
  @timeit to "Newton" while ((!converged) && (iter < maxiter))

    iter += 1

    # J = residualJacobian(V, Ybus, pv, pq)
    @timeit to "Jacobian" AD.residualJacobianAD!(jacobianAD, residualFunction_polar!, Vm, Va,
                        ybus_re, ybus_im, pbus, qbus, pv, pq, nbus, to)
    println("Preconditioner with $npartitions partitions")
    @timeit to "Preconditioner" P = Precondition.update(jacobianAD, preconditioner)
    J = jacobianAD.J
    if J isa SparseArrays.SparseMatrixCSC
      # @timeit to "Sparse solver" dx = -(J \ F)
      if solver == "bicgstab"
        @timeit to "CPU-BICGSTAB" (x, history) = IterativeSolvers.bicgstabl(P*J, P*F, log=true)
        push!(linsol_iters, history.iters)
      elseif solver == "gmres"
        @timeit to "CPU-GMRES" (x, history) = IterativeSolvers.gmres(P*J, P*F, log=true)
        push!(linsol_iters, history.iters)
      else
        error("Unknown solver $solver")
      end
      dx = -x
    end
    if J isa CuArrays.CUSPARSE.CuSparseMatrixCSR
      @timeit to "GPU-BICGSTAB" x, lin_iter = bicgstab(J, F, P, maxiter=500)
      push!(linsol_iters, lin_iter)
      # @timeit to "GPU-GMRES" (x, stats) = Krylov.dqgmres(J, F, M=P, memory=5, itmax=500)
      # @timeit to "Sparse solver" dx  = -CUSOLVER.csrlsvqr!(J,F,dx,lintol,one(Cint),'O')
      dx = -x
    end

    # update voltage
    if (npv != 0)
      Va[pv] .= Va[pv] .+ dx[j1:j2]
    end
    if (npq != 0)
      Va[pq] .= Va[pq] .+ dx[j3:j4]
      Vm[pq] .= Vm[pq] .+ dx[j5:j6]
    end

    V .= Vm .* exp.(1im .*Va)

    Vm = abs.(V)
    Va = Kernels.@angle(V)

    # evaluate residual and check for convergence
    # F = residualFunction(V, Ybus, Sbus, pv, pq)
    
    # v_re[:] = real(V)
    # v_im[:] = imag(V)
    #F .= 0.0
    #residualFunction_real!(F, v_re, v_im,
    #        ybus_re, ybus_im, pbus, qbus, pv, pq, nbus)
    
    F .= 0.0
    Kernels.@sync begin
    @timeit to "Residual function" Kernels.@dispatch threads=nthreads blocks=nblocks residualFunction_polar!(F, Vm, Va,
                          ybus_re.nzval, ybus_re.colptr, ybus_re.rowval, 
                          ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
                          pbus, qbus, pv, pq, nbus)
    end

    @timeit to "Norm" normF = norm(F, Inf)
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

  # Timer outputs display
  show(to)
  println("")
  reset_timer!(to)

  return V, converged, normF, linsol_iters[1], sum(linsol_iters)

end

# end of module
end
