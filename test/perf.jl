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
    pf = PowerSystem.PowerNetwork(datafile, 1)
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
    pf = PowerSystem.PowerNetwork(datafile, 1)
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

function run_level2(datafile; device=CPU())
    nlp, u = build_nlp(datafile, device)
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

function run_penalty(datafile; device=CPU())
    nlp, u = build_nlp(datafile, device)
    pen = ExaPF.PenaltyEvaluator(nlp)
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

datafile = joinpath(dirname(@__FILE__), "..", "data", "case9.m")

@info("ReducedSpaceEvaluator")
run_level2(datafile, device=CPU())
@info("PenaltyEvaluator")
run_penalty(datafile, device=CPU())
