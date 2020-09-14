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

const PS = PowerSystem

@testset "Compute reduced gradient on CPU" begin
    @testset "Case $case" for case in ["case9.m", "case30.m"]
        datafile = joinpath(dirname(@__FILE__), "data", case)
        tolerance = 1e-8
        pf = PS.PowerNetwork(datafile, 1)

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
        @testset "Reduced Jacobian" begin
            uk = copy(u)
            cons = ExaPF.power_constraints
            m = ExaPF.size_constraint(polar, cons)

            cons_x = (g_, x_) -> cons(polar, g_, x_, uk, p; V=eltype(x_))
            cons_u = (g_, u_) -> cons(polar, g_, xk, u_, p; V=eltype(u_))

            g = zeros(m)

            jx = ForwardDiff.jacobian(cons_x, g, xk)
            ju = ForwardDiff.jacobian(cons_u, g, uk)
        end
    end
end
