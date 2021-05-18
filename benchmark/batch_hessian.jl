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
        hessian_lagrangian = ExaPF.HessianLagrangian(polar, lufac, lufac')
    else
        gJ = CuSparseMatrixCSR(J)
        lufac = CUSOLVERRF.CusolverRfLU(gJ)
        blufac = CUSOLVERRF.CusolverRfLU(gJ)
        badjlu = CUSOLVERRF.CusolverRfLU(CuSparseMatrixCSC(J))
        linear_solver = LS.DirectSolver(lufac)
        hessian_lagrangian = ExaPF.HessianLagrangian(polar, blufac, badjlu)
    end

    nlp = @time ExaPF.ReducedSpaceEvaluator(polar; constraints=constraints,
                                            linear_solver=linear_solver,
                                            powerflow_solver=powerflow_solver,
                                            hessian_lagrangian=hessian_lagrangian)
    return nlp
end

function run_batch_hessian(nlp)
    # Update nlp to stay on manifold
    u = ExaPF.initial(nlp)
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

    hv = similar(u) ; fill!(hv, 0)
    v = similar(u) ; fill!(v, 0)
    v[1] = 1
    print("Hessprod \t")
    @btime ExaPF.hessprod_!($nlp, $hv, $u, $v)
    y = similar(nlp.g_min) ; fill!(y, 1.0)
    w = similar(nlp.g_min) ; fill!(w, 1.0)
    print("HLagPen-prod \t")
    @btime ExaPF.hessian_lagrangian_penalty_prod_!($nlp, $hv, $u, $y, 1.0, $v, $w)

    print("Hessian \t")
    n = ExaPF.n_variables(nlp)
    hess = similar(u, n, n)
    @time ExaPF.batch_hessian!(nlp, hess, u)
    @time ExaPF.batch_hessian!(nlp, hess, u)
    return
end

