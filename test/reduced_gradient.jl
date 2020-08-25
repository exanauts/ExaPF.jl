# Verify reduced gradient
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using SparseArrays

import ExaPF: ParseMAT, PowerSystem, IndexSet

@testset "RGM Optimal Power flow 9 bus case" begin
    datafile = "case9.m"
    pf = PowerSystem.PowerNetwork(datafile, 1)
    # retrieve initial state of network
    pbus = real.(pf.sbus)
    qbus = imag.(pf.sbus)
    vmag = abs.(pf.vbus)
    vang = angle.(pf.vbus)
    x = ExaPF.PowerSystem.get_x(pf, vmag, vang, pbus, qbus)
    u = ExaPF.PowerSystem.get_u(pf, vmag, vang, pbus, qbus)
    p = ExaPF.PowerSystem.get_p(pf, vmag, vang, pbus, qbus)

    # solve power flow
    xk, g, Jx, Ju, convergence, residualFunction_x! = ExaPF.solve(pf, x, u, p)
    ∇gₓ = Jx(pf, xk, u, p)
    ∇gᵤ = Ju(pf, xk, u, p)
    vecx = Vector{Float64}(undef, length(x) + length(u))
    vecx[1:length(x)] .= xk
    vecx[length(x)+1:end] .= u
    fjac = vecx -> ForwardDiff.jacobian(residualFunction_x!, vecx)
    jac = fjac(vecx)
    jacx = sparse(jac[:,1:length(x)])
    jacu = sparse(jac[:,length(x)+1:end])
    @test isapprox(∇gₓ, jacx)
    @test isapprox(∇gᵤ, jacu)
    hes = ForwardDiff.jacobian(fjac, vecx)
    # I am not sure about the reshape.
    # It could be that length(xk) goes at the end. This tensor stuff is a brain twister.
    hes = reshape(hes, (length(xk), length(xk) + length(u), length(xk) + length(u)))
    hesxx = hes[:, 1:length(x), 1:length(x)]
    hesxu = hes[:, 1:length(x), length(x)+1:end]
    hesuu = hes[:, length(x)+1:end, length(x)+1:end]


    c = ExaPF.cost_function(pf, xk, u, p)
    ∇fₓ, ∇fᵤ = ExaPF.cost_gradients(pf, xk, u, p)

    # Test gradients
    # We need uk here for the closure
    uk = copy(u)
    cost_x = x_ -> ExaPF.cost_function(pf, x_, uk, p; V=eltype(x_))
    cost_u = u_ -> ExaPF.cost_function(pf, xk, u_, p; V=eltype(u_))

    dCdx_fd = FiniteDiff.finite_difference_gradient(cost_x, xk)
    dCdx_ad = ForwardDiff.gradient(cost_x, xk)
    dCdu_fd = FiniteDiff.finite_difference_gradient(cost_u, u)
    dCdu_ad = ForwardDiff.gradient(cost_u, u)

    @test isapprox(∇fₓ, dCdx_fd)
    @test isapprox(∇fᵤ, dCdu_fd)
    @test isapprox(∇fₓ, dCdx_ad)
    @test isapprox(∇fᵤ, dCdu_ad)

    # residual function
    ybus_re, ybus_im = ExaPF.Spmat{Vector}(pf.Ybus)
    function g2(pf, x, u, p)
        eval_g = similar(x)
        nbus = length(pbus)
        Vm, Va, pbus, qbus = PowerSystem.retrieve_physics(pf, x, u, p)
        ExaPF.residualFunction_polar!(
            eval_g, Vm, Va,
            ybus_re, ybus_im,
            pbus, qbus, pf.pv, pf.pq, nbus
        )
        return eval_g
    end
    g_x = x_ -> g2(pf, x_, uk, p)
    ∇gₓ_fd = FiniteDiff.finite_difference_jacobian(g_x, xk)
    # This function should return the same matrix as ∇gₓ, but it
    # appears that is not the case
    @test isapprox(∇gₓ_fd, Array(∇gₓ))
    @info("m1: ", ∇gₓ_fd)
    @info("m2: ", Array(∇gₓ))

    g_u = u_ -> g2(pf, xk, u_, p)
    ∇gᵤ_fd = FiniteDiff.finite_difference_jacobian(g_u, uk)
    # However, it appears that the Jacobian wrt u is correct
    @test isapprox(∇gᵤ_fd, Array(∇gᵤ))
    @info("M: " ,Array(∇gᵤ))
    @info("M: " ,∇gᵤ_fd)

    # evaluate cost
    c = ExaPF.cost_function(pf, xk, uk, p)
    ## ADJOINT
    # lamba calculation
    λk  = -(∇gₓ') \ ∇fₓ
    grad_adjoint = ∇fᵤ + ∇gᵤ' * λk
    ## DIRECT
    S = - inv(Array(∇gₓ)) * ∇gᵤ
    grad_direct = ∇fᵤ + S' * ∇fₓ
    @test isapprox(grad_adjoint, grad_direct)

    function reduced_cost(u_)
        # Ensure we remain in the manifold
        x_, g, _, _, convergence = ExaPF.solve(pf, xk, u_, p, tol=1e-14)
        return ExaPF.cost_function(pf, x_, u_, p)
    end

    grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, uk)
    # At the end, we are unable to compute the reduced gradient
    @info("gd", grad_fd)
    @info("gd", grad_adjoint)
    @test isapprox(grad_fd, grad_adjoint)
end
