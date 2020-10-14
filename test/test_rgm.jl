# Verify solutions against matpower results
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using KernelAbstractions

import ExaPF: ParseMAT, PowerSystem, IndexSet

@testset "RGM Optimal Power flow 9 bus case" begin
    datafile = joinpath(dirname(@__FILE__), "data", "case9.m")
    pf = PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, CPU())

    xk = ExaPF.initial(polar, State())
    uk = ExaPF.initial(polar, Control())
    p = ExaPF.initial(polar, Parameters())

    constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
    nlp = ExaPF.ReducedSpaceEvaluator(polar, xk, uk, p; constraints=constraints)

    # solve power flow
    ExaPF.update!(nlp, uk)

    # reduced gradient method
    iterations = 0
    iter_max = 100
    step = 0.0001
    norm_grad = 10000
    norm_tol = 1e-5

    iter = 1

    # initial gradient
    grad = similar(uk)
    wk = similar(uk)
    up = copy(uk)
    fill!(grad, 0)

    while norm_grad > norm_tol && iter < iter_max
        ExaPF.update!(nlp, uk; verbose_level=ExaPF.VERBOSE_LEVEL_NONE)
        c = ExaPF.objective(nlp, uk)
        ExaPF.gradient!(nlp, grad, uk)
        # compute control step
        wk = uk - step*grad
        ExaPF.project_constraints!(nlp, uk, wk)
        norm_grad = norm(uk .- up, Inf)
        iter += 1
        up .= uk
    end
    @test iter == 39
    @test isapprox(uk, [1.1, 1.343109921105559, 0.9421135274454701, 1.1, 1.1], atol=1e-4)
end

