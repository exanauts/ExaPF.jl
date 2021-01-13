using CUDA
using CUDA.CUSPARSE
using ExaPF
using KernelAbstractions
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using TimerOutputs
import ExaPF: PowerSystem, IndexSet, AutoDiff

function run_level1(datafile; device=CPU())
    pf = PowerSystem.PowerNetwork(datafile)
    polar = PolarForm(pf, device)
    cache = ExaPF.NetworkState(polar)
    jx, ju, ∂obj = ExaPF.init_autodiff_factory(polar, cache)

    @btime begin
        cache = ExaPF.NetworkState($polar)
        ExaPF.powerflow($polar, $jx, cache, tol=1e-14, verbose_level=0)
    end
    return
end

function build_nlp(datafile, device)
    pf = PowerSystem.PowerNetwork(datafile)
    polar = PolarForm(pf, device)
    x0 = ExaPF.initial(polar, State())
    u0 = ExaPF.initial(polar, Control())
    p = ExaPF.initial(polar, Parameters())

    constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
    print("Constructor\t")
    nlp = @time ExaPF.ReducedSpaceEvaluator(polar, x0, u0, p; constraints=constraints ,
                                            ε_tol=1e-10)
    return nlp, u0
end

function run_level2(nlp, u; device=CPU())
    # Update nlp to stay on manifold
    print("Update   \t")
    @time ExaPF.update!(nlp, u)
    # Compute objective
    print("Objective\t")
    c = @time ExaPF.objective(nlp, u)
    # Compute gradient of objective
    g = similar(u)
    fill!(g, 0)
    print("Gradient \t")
    @time ExaPF.gradient!(nlp, g, u)

    # Constraint
    ## Evaluation of the constraints
    cons = similar(nlp.g_min)
    fill!(cons, 0)
    print("Constrt \t")
    @time ExaPF.constraint!(nlp, cons, u)

    print("Jac-prod \t")
    jv = copy(g) ; fill!(jv, 0)
    v = copy(cons) ; fill!(v, 1)
    @time ExaPF.jtprod!(nlp, jv, u, v)
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

datafile = joinpath(dirname(@__FILE__), "..", "data", "case9.m")
# device = CPU()
# nlp, u = build_nlp(datafile, CPU())

@info("ReducedSpaceEvaluator")
run_level2(nlp, u, device=CPU())
ExaPF.reset!(nlp)
@info("AugLagEvaluator")
run_penalty(nlp, u, device=CPU())
ExaPF.reset!(nlp)
@info("ProxALEvaluator")
run_proxal(nlp, u, device=CPU())
ExaPF.reset!(nlp)
@info("ProxAugEvaluator")
run_proxaug(nlp, u, device=CPU())
ExaPF.reset!(nlp)

