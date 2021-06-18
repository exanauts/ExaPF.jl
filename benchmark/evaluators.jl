using LinearAlgebra
using SparseArrays
using KernelAbstractions
using BenchmarkTools

using ExaPF
import ExaPF: LinearSolvers

const LS = LinearSolvers

function build_nlp(datafile, device)
    print("Load data\t")
    polar = @time PolarForm(datafile, device)

    constraints = Function[
        ExaPF.voltage_magnitude_constraints,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
    ]
    print("Constructor\t")
    powerflow_solver = NewtonRaphson(tol=1e-10)
    nlp = @time ExaPF.ReducedSpaceEvaluator(polar; constraints=constraints,
                                            powerflow_solver=powerflow_solver)
    return nlp
end

function run_reduced_evaluator(nlp, u; device=CPU())
    n = length(u)
    # Update nlp to stay on manifold
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

    # Constraint
    ## Evaluation of the constraints
    cons = similar(nlp.g_min)
    fill!(cons, 0)
    print("Constrt \t")
    @btime ExaPF.constraint!($nlp, $cons, $u)

    print("Jac-prod \t")
    jv = copy(g) ; fill!(jv, 0)
    v = copy(cons) ; fill!(v, 1)
    @btime ExaPF.jtprod!($nlp, $jv, $u, $v)
    hv = similar(u) ; fill!(hv, 0)
    v = similar(u) ; fill!(v, 0)
    v[1] = 1
    print("Hessprod \t")
    @btime ExaPF.hessprod!($nlp, $hv, $u, $v)
    y = similar(cons) ; fill!(y, 1.0)
    w = similar(cons) ; fill!(w, 1.0)
    print("HLagPen-prod \t")
    @btime ExaPF.hessian_lagrangian_penalty_prod!($nlp, $hv, $u, $y, 1.0, $w, $v)
    print("Hessian \t")
    H = similar(u, n, n)
    @btime ExaPF.hessian!($nlp, $H, $u)
    @btime ExaPF.batch_hessian!($nlp, $H, $u)
    return
end

function run_slack(nlp, u; device=CPU())
    slack = ExaPF.SlackEvaluator(nlp)
    w = ExaPF.initial(slack)
    print("Update   \t")
    @time ExaPF.update!(slack, w)
    print("Objective\t")
    c = @time ExaPF.objective(slack, w)
    g = similar(w)
    fill!(g, 0)
    print("Gradient \t")
    @time ExaPF.gradient!(slack, g, w)
    cons = similar(nlp.g_min)
    fill!(cons, 0)
    print("Constrt \t")
    @time ExaPF.constraint!(slack, cons, w)

    hv = similar(w) ; fill!(hv, 0)
    v = similar(w) ;  fill!(v, 1)
    print("Hessprod \t")
    @time ExaPF.hessprod!(slack, hv, w, v)
    y = similar(cons) ; fill!(y, 1.0)
    z = similar(cons) ; fill!(w, 1.0)
    print("HLagPen-prod \t")
    @time ExaPF.hessian_lagrangian_penalty_prod!(slack, hv, w, y, 1.0, v, z)
    return
end

function run_penalty(nlp, u; device=CPU())
    pen = ExaPF.AugLagEvaluator(nlp, u)
    print("Update   \t")
    @time ExaPF.update!(pen, u)
    print("Objective\t")
    c = @time ExaPF.objective(pen, u)
    g = similar(u)
    fill!(g, 0)
    print("Gradient \t")
    @time ExaPF.gradient!(pen, g, u)
    hv = similar(u) ; fill!(hv, 0)
    v = similar(u) ;  fill!(v, 1)
    print("Hessprod \t")
    @time ExaPF.hessprod!(pen, hv, u, v)
    return
end

function run_slackaug(nlp, u; device=CPU())
    nlp2 = ExaPF.SlackEvaluator(nlp)
    w = ExaPF.initial(nlp2)
    pen = ExaPF.AugLagEvaluator(nlp2, w)
    print("Update   \t")
    @time ExaPF.update!(pen, w)
    print("Objective\t")
    c = @time ExaPF.objective(pen, w)
    g = similar(w)
    fill!(g, 0)
    print("Gradient \t")
    @time ExaPF.gradient!(pen, g, w)
    hv = similar(w) ; fill!(hv, 0)
    v = similar(w) ;  fill!(v, 1)
    print("Hessprod \t")
    @time ExaPF.hessprod!(pen, hv, w, v)
    return
end

function run_proxal(nlp, u; device=CPU())
    pen = ExaPF.ProxALEvaluator(nlp, ExaPF.Normal)
    w = ExaPF.initial(pen)
    print("Update   \t")
    @time ExaPF.update!(pen, w)
    print("Objective\t")
    c = @time ExaPF.objective(pen, w)
    g = similar(w)
    fill!(g, 0)
    print("Gradient \t")
    @time ExaPF.gradient!(pen, g, w)
    return
end

function run_proxaug(nlp, u; device=CPU())
    pen = ExaPF.ProxALEvaluator(nlp, ExaPF.Normal)
    w = ExaPF.initial(pen)
    aug = ExaPF.AugLagEvaluator(pen, w)
    print("Update   \t")
    @time ExaPF.update!(aug, w)
    print("Objective\t")
    c = @time ExaPF.objective(aug, w)
    g = similar(w)
    fill!(g, 0)
    print("Gradient \t")
    @time ExaPF.gradient!(aug, g, w)
    return
end

datafile = joinpath(dirname(@__FILE__), "..", "data", "case300.m")
device = CPU()
nlp = build_nlp(datafile, CPU())
u = ExaPF.initial(nlp)

@info("ReducedSpaceEvaluator")
run_reduced_evaluator(nlp, u, device=CPU())
ExaPF.reset!(nlp)
# @info("SlackEvaluator")
# run_slack(nlp, u, device=CPU())
# ExaPF.reset!(nlp)
# @info("AugLagEvaluator")
# run_penalty(nlp, u, device=CPU())
# ExaPF.reset!(nlp)
# @info("SlackAugLagEvaluator")
# run_slackaug(nlp, u, device=CPU())
# ExaPF.reset!(nlp)
