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
        cache = ExaPF.NetworkState(polar)

        xk = ExaPF.initial(polar, State())
        u = ExaPF.initial(polar, Control())
        p = ExaPF.initial(polar, Parameters())

        jx, ju, ∂obj = ExaPF.init_ad_factory(polar, cache)

        # solve power flow
        conv = powerflow(polar, jx, cache, tol=1e-12)
        ExaPF.get!(polar, State(), xk, cache)
        # No need to recompute ∇gₓ
        ∇gₓ = jx.J
        ∇gᵤ = ExaPF.jacobian(polar, ju, cache)
        # test jacobian wrt x
        ∇gᵥ = ExaPF.jacobian(polar, jx, cache)
        @test isapprox(∇gₓ, ∇gᵥ)

        # Test with Matpower's Jacobian
        V = cache.vmag .* exp.(im * cache.vang)
        Ybus = pf.Ybus
        J = ExaPF.residualJacobian(V, Ybus, pf.pv, pf.pq)
        @test isapprox(∇gₓ, J)

        # Test gradients
        @testset "Reduced gradient" begin
            # Refresh cache with new values of vmag and vang
            ExaPF.refresh!(polar, PS.Generator(), PS.ActivePower(), cache)
            # We need uk here for the closure
            uk = copy(u)
            ExaPF.cost_production_adjoint(polar, ∂obj, cache)
            ∇fₓ = ∂obj.∇fₓ
            ∇fᵤ = ∂obj.∇fᵤ

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
                ExaPF.transfer!(polar, cache, xk, u_, p)
                convergence = powerflow(polar, jx, cache, tol=1e-14)
                ExaPF.refresh!(polar, PS.Generator(), PS.ActivePower(), cache)
                return ExaPF.cost_production(polar, cache.pg)
            end

            grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, uk)
            @test isapprox(grad_fd, grad_adjoint, rtol=1e-4)
        end
    end
end
