using CUDA
using CUDA.CUSPARSE
using Revise
using Printf
using FiniteDiff
using ForwardDiff
using BenchmarkTools
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs
using ExaPF
import ExaPF: PowerSystem, AD, Precondition, Iterative

@testset "Formulation" begin
    datafile = "test/data/case9.m"
    tolerance = 1e-8
    pf = PowerSystem.PowerNetwork(datafile, 1)

    @testset "Device $device" for (device, M) in zip([CPU(), CUDADevice()], [Array, CuArray])
        polar = PolarForm(pf, device)

        b = bounds(polar, State())
        b = bounds(polar, Control())

        # Test getters
        nᵤ = ExaPF.get(polar, NumberOfControl())
        nₓ = ExaPF.get(polar, NumberOfState())

        # Get initial position
        x0 = ExaPF.initial(polar, State())
        u0 = ExaPF.initial(polar, Control())
        p = ExaPF.initial(polar, Parameters())

        @test length(u0) == nᵤ
        @test length(x0) == nₓ
        for v in [x0, u0, p]
            @test isa(v, M)
        end

        # Init AD factory
        jx, ju = ExaPF.init_ad_factory(polar, x0, u0, p)

        # Test powerflow
        xₖ, _ = @time powerflow(polar, jx, x0, u0, p, verbose_level=0, tol=tolerance)

        # Test callbacks
        ## Power Balance
        g = ExaPF.power_balance(polar, xₖ, u0, p)
        @test isa(g, M)
        # As we run powerflow before, the balance should be below tolerance
        @test norm(g, Inf) < tolerance
        ## Cost Production
        c = @time ExaPF.cost_production(polar, xₖ, u0, p)
        @test isa(c, Real)
        ## Inequality constraint
        for cons in [ExaPF.state_constraint, ExaPF.power_constraints]
            m = ExaPF.size_constraint(polar, cons)
            g = M{Float64, 1}(undef, m) # TODO: this signature is not great
            fill!(g, 0)
            cons(polar, g, xₖ, u0, p)
        end
    end

    @testset "Test AD on CPU" begin
        polar = PolarForm(pf, CPU())

        x0 = ExaPF.initial(polar, State())
        u0 = ExaPF.initial(polar, Control())
        p = ExaPF.initial(polar, Parameters())

        jx, ju = ExaPF.init_ad_factory(polar, x0, u0, p)

        # solve power flow
        xk, conv = powerflow(polar, jx, x0, u0, p, tol=1e-12)
        # No need to recompute ∇gₓ
        ∇gₓ = jx.J
        ∇gᵤ = ExaPF.jacobian(polar, ju, xk, u0, p)
        # Test Jacobian wrt x
        ∇gᵥ = ExaPF.jacobian(polar, jx, xk, u0, p)
        @test isapprox(∇gₓ, ∇gᵥ)

        function residualFunction_x!(vecx)
            nx = ExaPF.get(polar, NumberOfState())
            nu = ExaPF.get(polar, NumberOfControl())
            x_ = Vector{eltype(vecx)}(undef, nx)
            u_ = Vector{eltype(vecx)}(undef, nu)
            x_ .= vecx[1:length(x)]
            u_ .= vecx[length(x)+1:end]
            g = ExaPF.power_balance(polar, x_, u_, p; V=eltype(x_))
            return g
        end

        x, u = xk, u0
        vecx = Vector{Float64}(undef, length(x) + length(u))
        vecx[1:length(x)] .= x
        vecx[length(x)+1:end] .= u
        fjac = vecx -> ForwardDiff.jacobian(residualFunction_x!, vecx)
        jac = fjac(vecx)
        jacx = sparse(jac[:,1:length(x)])
        jacu = sparse(jac[:,length(x)+1:end])
        @test isapprox(∇gₓ, jacx, rtol=1e-5)
        @test isapprox(∇gᵤ, jacu, rtol=1e-5)

        # Test gradients
        @testset "Reduced gradient" begin
            # We need uk here for the closure
            uk = copy(u)
            cost_x = x_ -> ExaPF.cost_production(polar, x_, uk, p; V=eltype(x_))
            cost_u = u_ -> ExaPF.cost_production(polar, xk, u_, p; V=eltype(u_))
            ∇fₓ = ForwardDiff.gradient(cost_x, xk)
            ∇fᵤ = ForwardDiff.gradient(cost_u, uk)

            ## ADJOINT
            # lamba calculation
            λk  = -(∇gₓ') \ ∇fₓ
            grad_adjoint = ∇fᵤ + ∇gᵤ' * λk
            # ## DIRECT
            S = - inv(Array(∇gₓ)) * ∇gᵤ
            grad_direct = ∇fᵤ + S' * ∇fₓ
            @test isapprox(grad_adjoint, grad_direct)

            # Compare with finite difference
            function reduced_cost(u_)
                # Ensure we remain in the manifold
                x_, convergence = powerflow(polar, jx, xk, u_, p, tol=1e-14)
                return ExaPF.cost_production(polar, x_, u_, p)
            end

            grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, uk)
            @test isapprox(grad_fd, grad_adjoint, rtol=1e-4)
        end
    end
end
