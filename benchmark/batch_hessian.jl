using LinearAlgebra
using SparseArrays
using KernelAbstractions
using BenchmarkTools

using ExaPF
import ExaPF: LinearSolvers

const LS = LinearSolvers

function build_batch_nlp(datafile, device, nbatch)
    print("Load data\t")
    polar = @time PolarForm(datafile, device)

    constraints = Function[
        ExaPF.voltage_magnitude_constraints,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
    ]
    powerflow_solver = NewtonRaphson(tol=1e-10)

    J = ExaPF.jacobian_sparsity(polar, ExaPF.power_balance, State())
    if isa(device, CPU)
        lufac = lu(J)
        linear_solver = LS.DirectSolver(lufac)
        if nbatch == 1
            hessian_lagrangian = ExaPF.HessianLagrangian(polar, lufac, lufac')
        else
            hessian_lagrangian = ExaPF.BatchHessianLagrangian(polar, lufac, lufac', nbatch)
        end
    else
        gJ = CuSparseMatrixCSR(J)
        lufac = CUSOLVERRF.CusolverRfLU(gJ)
        linear_solver = LS.DirectSolver(lufac)
        if nbatch == 1
            blufac = CUSOLVERRF.CusolverRfLU(gJ)
            badjlu = CUSOLVERRF.CusolverRfLU(CuSparseMatrixCSC(J))
            hessian_lagrangian = ExaPF.HessianLagrangian(polar, blufac, badjlu)
        else
            blufac = CUSOLVERRF.CusolverRfLUBatch(gJ, nbatch)
            badjlu = CUSOLVERRF.CusolverRfLUBatch(CuSparseMatrixCSC(J), nbatch)
            hessian_lagrangian = ExaPF.BatchHessianLagrangian(polar, blufac, badjlu, nbatch)
        end
    end

    nlp = @time ExaPF.ReducedSpaceEvaluator(polar; constraints=constraints,
                                            linear_solver=linear_solver,
                                            powerflow_solver=powerflow_solver,
                                            hessian_lagrangian=hessian_lagrangian)
    return nlp
end

function run_batch_hessian(nlp)
    @assert !isnothing(nlp.hesslag)
    nbatch = ExaPF.n_batches(nlp.hesslag)
    # Update nlp to stay on manifold
    u = ExaPF.initial(nlp)
    n = ExaPF.n_variables(nlp)
    print("Update   \t")
    @time ExaPF.update!(nlp, u)
    # Compute objective
    print("Objective\t")
    c = @btime ExaPF.objective($nlp, $u)
    # Compute gradient of objective
    g = similar(u)
    fill!(g, 0)
    print("Gradient \t")
    @btime ExaPF.gradient!($nlp, $g, $u)

    if nbatch == 1
        hv = similar(u) ; fill!(hv, 0)
        v = similar(u) ; fill!(v, 0)
        v[1] = 1.0
    else
        hv = similar(u, n, nbatch) ; fill!(hv, 0)
        v = similar(u, n, nbatch) ; fill!(v, 0)
        v[1, :] .= 1.0
    end
    print("Hessprod \t")
    @btime ExaPF.hessprod_!($nlp, $hv, $u, $v)
    y = similar(nlp.g_min) ; fill!(y, 1.0)
    w = similar(nlp.g_min) ; fill!(w, 1.0)
    print("HLagPen-prod \t")
    @btime ExaPF.hessian_lagrangian_penalty_prod_!($nlp, $hv, $u, $y, 1.0, $v, $w)
    return
end

