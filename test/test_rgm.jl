# Verify solutions against matpower results
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra

# Include the linesearch here for now
include("../src/algorithms/linesearches.jl")

import ExaPF: ParseMAT, PowerSystem, IndexSet

@testset "RGM Optimal Power flow 9 bus case" begin
    datafile = joinpath(dirname(@__FILE__), "data", "case9.m")
    pf = PowerSystem.PowerNetwork(datafile, 1)

    # retrieve initial state of network
    pbus = real.(pf.sbus)
    qbus = imag.(pf.sbus)
    vmag = abs.(pf.vbus)
    vang = angle.(pf.vbus)

    x = ExaPF.PowerSystem.get_x(pf, vmag, vang, pbus, qbus)
    u = ExaPF.PowerSystem.get_u(pf, vmag, vang, pbus, qbus)
    p = ExaPF.PowerSystem.get_p(pf, vmag, vang, pbus, qbus)
    u_min, u_max, x_min, x_max = ExaPF.get_constraints(pf)

    # solve power flow
    xk, g, Jx, Ju, convergence = ExaPF.solve(pf, x, u, p)
    dGdx = Jx(pf, x, u, p)
    dGdu = Ju(pf, x, u, p)

    c = ExaPF.cost_function(pf, xk, u, p)
    dCdx, dCdu = ExaPF.cost_gradients(pf, xk, u, p)

    # Test gradients
    # We need uk here for the closure
    uk = copy(u)
    function cost_x(xk)
        return ExaPF.cost_function(pf, xk, uk, p; V=eltype(xk))
    end

    function cost_u(uk)
        return ExaPF.cost_function(pf, xk, uk, p; V=eltype(uk))
    end

    dCdx_fd = FiniteDiff.finite_difference_gradient(cost_x,xk)
    dCdx_ad = ForwardDiff.gradient(cost_x,xk)
    dCdu_fd = FiniteDiff.finite_difference_gradient(cost_u,u)
    dCdu_ad = ForwardDiff.gradient(cost_u,u)

    @test isapprox(dCdx,dCdx_fd)
    @test isapprox(dCdu,dCdu_fd)
    @test isapprox(dCdx,dCdx_ad)
    @test isapprox(dCdu,dCdu_ad)

    # reduced gradient method
    iterations = 0
    iter_max = 100
    step = 0.0001
    norm_grad = 10000
    norm_tol = 1e-5

    iter = 1
    while norm_grad > norm_tol && iter < iter_max
        println("Iteration: ", iter)

        # solve power flow and compute gradients
        xk, g, Jx, Ju, convergence = ExaPF.solve(pf, xk, uk, p)
        dGdx = Jx(pf, xk, uk, p)
        dGdu = Ju(pf, xk, uk, p)
        fdCdx = xk -> ForwardDiff.gradient(cost_x,xk)
        fdCdu = uk -> ForwardDiff.gradient(cost_u,uk)
        dCdx = fdCdx(xk)
        dCdu = fdCdu(uk)

        # evaluate cost
        c = ExaPF.cost_function(pf, xk, uk, p)

        # lamba calculation
        lambda = -(dGdx'\dCdx)

        # Form functions
        Lu = u -> ExaPF.cost_function(pf, xk, u, p) + (g(pf, xk, u, p))'*lambda
        grad_Lu = u -> fdCdu(u) + (Ju(pf, xk, u, p)')*lambda
        # compute gradient
        grad = grad_Lu(uk)
        println("Cost: ", c)
        println("Norm: ", norm(grad))
        # Optional linesearch
        # step = ls(uk, grad, Lu, grad_Lu)
        # compute control step
        uk = uk - step*grad
        ExaPF.project_constraints!(uk, grad, u_min, u_max)
        println("Gradient norm: ", norm(grad))
        norm_grad = norm(grad)

        iter += 1
    end
    ExaPF.PowerSystem.print_state(pf, xk, uk, p)

end
