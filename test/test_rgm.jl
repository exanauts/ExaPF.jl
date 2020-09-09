# Verify solutions against matpower results
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using KernelAbstractions

# Include the linesearch here for now
include("../src/algorithms/linesearches.jl")

import ExaPF: ParseMAT, PowerSystem, IndexSet

@testset "RGM Optimal Power flow 9 bus case" begin
    datafile = joinpath(dirname(@__FILE__), "data", "case9.m")
    pf = PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, CPU())
    
    xk = ExaPF.initial(polar, State())
    uk = ExaPF.initial(polar, Control())
    p = ExaPF.initial(polar, Parameters())
    
    constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
    nlp = @time ExaPF.ReducedSpaceEvaluator(polar, xk, uk, p; constraints=constraints)

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
    fill!(grad, 0)

    while norm_grad > norm_tol && iter < iter_max
        println("Iteration: ", iter)

        # solve power flow and compute gradients
        ExaPF.update!(nlp, uk)

        # evaluate cost
        c = ExaPF.objective(nlp, uk)

        # compute gradient
        ExaPF.gradient!(nlp, grad, uk)
        
        println("Cost: ", c)
        println("Norm: ", norm(grad))
        
        # Optional linesearch
        # step = ls(uk, grad, Lu, grad_Lu)
        # compute control step
        uk = uk - step*grad
        
        ExaPF.project_constraints!(uk, grad, nlp.u_min, nlp.u_max)
        println("Gradient norm: ", norm(grad))
        norm_grad = norm(grad)

        iter += 1
    end
    ExaPF.PowerSystem.print_state(pf, nlp.x, uk, p)

end
