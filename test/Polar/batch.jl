using Printf
using CUDA
using CUDA.CUSPARSE
using FiniteDiff
using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs
using ExaPF
import ExaPF: PowerSystem, AutoDiff

const LS = ExaPF.LinearSolvers
const PS = PowerSystem

## Uncomment once cusolverRF merged in CUDA.jl
# function LS.batch_ldiv!(lu_fac, DX, Js::ExaPF.BatchCuSparseMatrixCSR, F)
#     nbatch = length(Js)
#     CUSOLVERRF.rf_batch_refactor!(lu_fac, Js.nzVal)
#     ldiv!(DX, lu_fac, F)
# end
#

function test_batch_powerflow(polar, device, M)

    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())

    nbatch = 16
    b_cache = ExaPF.batch_buffer(polar, nbatch)

    # Test power balance in batch mode
    cons = b_cache.balance
    ExaPF.power_balance(polar, cons, b_cache)

    # Test batch Jacobian
    jx = ExaPF.BatchJacobian(polar, ExaPF.power_balance, State(), nbatch)
    ExaPF.batch_jacobian!(polar, jx, b_cache)

    ## TODO
    # Batch powerflow needs cusolverRF on the GPU. Currently not supported.
    isa(device, GPU) && return

    if isa(device, CPU)
        lu_fac = lu(jx.J[1])
        linear_solver = LS.DirectSolver(lu_fac)
    else
        jac = ExaPF.jacobian_sparsity(polar, ExaPF.power_balance, State())
        gjac = CuSparseMatrixCSR(jac)
        lu_fac = CUSOLVERRF.CusolverRfLUBatch(gjac, nbatch)
        linear_solver = LS.DirectSolver(lu_fac)
    end
    conv = ExaPF.batch_powerflow(polar, jx, b_cache, NewtonRaphson(), linear_solver)
    @test conv.has_converged
    return
end

function test_batch_hessian(polar, device, VT; nbatch=64)
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    nx = get(polar, NumberOfState())
    nu = get(polar, NumberOfControl())

    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)

    λ = ones(nx) |> VT
    tgt = rand(nx + nu) |> VT
    projp = zeros(nx + nu) |> VT

    # Evaluate Hessian-vector product (full ∇²gₓₓ is a 3rd dimension tensor)
    ∇²gλ = ExaPF.matpower_hessian(polar, ExaPF.power_balance, cache, λ)

    single_H = AutoDiff.Hessian(polar, ExaPF.power_balance)
    AutoDiff.adj_hessian_prod!(polar, single_H, projp, cache, λ, tgt)

    batch_H = ExaPF.BatchHessian(polar, ExaPF.power_balance, nbatch)

    MT = isa(device, GPU) ? CuMatrix : Matrix

    batch_tgt = rand(nx + nu, nbatch) |> MT
    batch_projp = zeros(nx + nu, nbatch) |> MT

    H = [
        ∇²gλ.xx  ∇²gλ.xu' ;
        ∇²gλ.xu  ∇²gλ.uu
    ]

    ExaPF.update_hessian!(polar, batch_H, cache)
    ExaPF.batch_adj_hessian_prod!(polar, batch_H, batch_projp, cache, λ, batch_tgt)

    if !isa(device, GPU)
        @test isapprox(batch_projp, H * batch_tgt)
    end
end

