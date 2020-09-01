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
include("../src/models/models.jl")

@testset "Formulation" begin
    datafile = "test/data/case9.m"
    tolerance = 1e-8
    pf = PowerSystem.PowerNetwork(datafile, 1)

    @testset "Device $device" for (device, M) in zip([CPU(), CUDADevice()], [Array, CuArray])
        polar = PolarForm(pf, device)

        b = bounds(polar, State())
        b = bounds(polar, Control())

        nᵤ = get(polar, NumberOfControl())
        nₓ = get(polar, NumberOfState())

        # Get initial position
        x0 = initial(polar, State())
        u0 = initial(polar, Control())
        p = initial(polar, Parameters())

        @test length(u0) == nᵤ
        @test length(x0) == nₓ
        for v in [x0, u0, p]
            @test isa(v, M)
        end

        # Init AD factory
        jx, ju = init_ad_factory(polar, x0, u0, p)

        # Test powerflow
        @time powerflow(polar, jx, x0, u0, p)
        xₖ, _ = @time powerflow(polar, jx, x0, u0, p, verbose_level=0, tol=tolerance)

        # Test callbacks
        g = power_balance(polar, xₖ, u0, p)
        c = cost_production(polar, xₖ, u0, p)
        @test isa(c, Real)
        @test norm(g, Inf) < tolerance
    end

    @testset "Test AD on CPU" begin
        polar = PolarForm(pf, CPU())

        x0 = initial(polar, State())
        u0 = initial(polar, Control())
        p = initial(polar, Parameters())

        jx, ju = init_ad_factory(polar, x0, u0, p)

        # solve power flow
        xk, conv = powerflow(polar, jx, x0, u0, p, tol=1e-12)
        # No need to recompute ∇gₓ
        ∇gₓ = jx.J
        ∇gᵤ = jacobian(polar, ju, xk, u0, p)
        # Test Jacobian wrt x
        ∇gᵥ = jacobian(polar, jx, xk, u0, p)
        @test isapprox(∇gₓ, ∇gᵥ)

        function residualFunction_x!(vecx)
            nx = get(polar, NumberOfState())
            nu = get(polar, NumberOfControl())
            x_ = Vector{eltype(vecx)}(undef, nx)
            u_ = Vector{eltype(vecx)}(undef, nu)
            x_ .= vecx[1:length(x)]
            u_ .= vecx[length(x)+1:end]
            g = power_balance(polar, x_, u_, p; V=eltype(x_))
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
        # @info("j", Array(∇gᵤ))
        # @info("j", Array(jacu))
        @test isapprox(∇gₓ, jacx, rtol=1e-5)
        @test_broken isapprox(∇gᵤ, jacu, rtol=1e-5)

        # Test gradients
        @testset "Reduced gradient" begin
            # We need uk here for the closure
            uk = copy(u)
            cost_x = x_ -> cost_production(polar, x_, uk, p; V=eltype(x_))
            cost_u = u_ -> cost_production(polar, xk, u_, p; V=eltype(u_))
            ∇fₓ = ForwardDiff.gradient(cost_x, xk)
            ∇fᵤ = ForwardDiff.gradient(cost_u, uk)

            ## ADJOINT
            # lamba calculation
            λk  = -(∇gₓ') \ ∇fₓ
            grad_adjoint = ∇fᵤ + jacu' * λk
            # ## DIRECT
            S = - inv(Array(∇gₓ)) * jacu
            grad_direct = ∇fᵤ + S' * ∇fₓ
            @test isapprox(grad_adjoint, grad_direct)

            # Compare with finite difference
            function reduced_cost(u_)
                # Ensure we remain in the manifold
                x_, convergence = powerflow(polar, jx, xk, u_, p, tol=1e-14)
                return cost_production(polar, x_, u_, p)
            end

            grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, uk)
            @test isapprox(grad_fd, grad_adjoint, rtol=1e-4)
        end
    end
end
