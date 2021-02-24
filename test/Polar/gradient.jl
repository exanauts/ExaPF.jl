using Printf
using FiniteDiff
using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs
using ExaPF
import ExaPF: PowerSystem, AutoDiff

const PS = PowerSystem

@testset "Compute reduced gradient on CPU" begin
    @testset "Case $case" for case in ["case9.m", "case30.m"]
        datafile = joinpath(dirname(@__FILE__), "..", "..", "data", case)
        tolerance = 1e-8
        pf = PS.PowerNetwork(datafile)

        polar = PolarForm(pf, CPU())
        cache = ExaPF.get(polar, ExaPF.PhysicalState())
        ExaPF.init_buffer!(polar, cache)

        xk = ExaPF.initial(polar, State())
        u = ExaPF.initial(polar, Control())

        jx, ju, ∂obj = ExaPF.init_autodiff_factory(polar, cache)

        # solve power flow
        conv = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-12))
        # No need to recompute ∇gₓ
        ∇gₓ = jx.J
        ∇gᵤ = ExaPF.jacobian(polar, ju, cache, AutoDiff.ControlJacobian())
        # test jacobian wrt x
        ∇gᵥ = ExaPF.jacobian(polar, jx, cache, AutoDiff.StateJacobian())
        @test isapprox(∇gₓ, ∇gᵥ)

        # Test with Matpower's Jacobian
        V = cache.vmag .* exp.(im * cache.vang)
        Ybus = pf.Ybus
        J = ExaPF.residual_jacobian(State(), V, Ybus, pf.pv, pf.pq, pf.ref)
        @test isapprox(∇gₓ, J)

        # Test gradients
        @testset "Reduced gradient" begin
            # Refresh cache with new values of vmag and vang
            ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
            # We need uk here for the closure
            uk = copy(u)
            ExaPF.∂cost(polar, ∂obj, cache)
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
                ExaPF.transfer!(polar, cache, u_)
                convergence = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-14))
                ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
                return ExaPF.cost_production(polar, cache.pg)
            end

            grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, uk)
            @test isapprox(grad_fd, grad_adjoint, rtol=1e-4)
        end
        @testset "Reduced Jacobian" begin
            for cons in [
                ExaPF.voltage_magnitude_constraints,
                ExaPF.active_power_constraints,
                ExaPF.reactive_power_constraints
            ]
                m = ExaPF.size_constraint(polar, cons)
                for icons in 1:m
                    ExaPF.jacobian(polar, cons, icons, ∂obj, cache)
                end
            end
        end
        # We test apart this routine, as it depends on Zygote
        @testset "Gradient of line-flow constraints" begin
            # Adjoint of flow_constraints()
            nbus = length(cache.vmag)
            M = typeof(u)
            m = ExaPF.size_constraint(polar, ExaPF.flow_constraints)
            x = ExaPF.xzeros(M, 2 * nbus)
            x[1:nbus] .= cache.vmag
            x[1+nbus:2*nbus] .= cache.vang
            bus_gen = polar.indexing.index_generators
            VI = typeof(bus_gen)

            ## Example with using sum as a sort of lumping of all constraints
            function lumping(x)
                VT = typeof(x)
                # Needed for ForwardDiff to have a cache with the right active type VT
                adcache = ExaPF.PolarNetworkState{VI, VT}(
                    cache.vmag, cache.vang, cache.pinj, cache.qinj,
                    cache.pg, cache.qg, cache.balance, cache.dx, bus_gen,
                )
                adcache.vmag .= x[1:nbus]
                adcache.vang .= x[1+nbus:2*nbus]
                g = VT(undef, m) ; fill!(g, 0)
                ExaPF.flow_constraints(polar, g, adcache)
                return sum(g)
            end
            adgradg = ForwardDiff.gradient(lumping,x)
            fdgradg = FiniteDiff.finite_difference_gradient(lumping,x)
            ## We pick sum() as the reduction function. This could be a mask function for active set or some log(x) for lumping.
            m_flows = ExaPF.size_constraint(polar, ExaPF.flow_constraints)
            weights = ones(m_flows)
            zygradg = ExaPF.xzeros(M, 2 * nbus)
            ExaPF.flow_constraints_grad!(polar, zygradg, cache, weights)
            # Verify  ForwardDiff and Zygote agree on the gradient
            @test isapprox(adgradg, fdgradg)
            # This breaks because of issue #89
            @test_broken isapprox(adgradg, zygradg)
        end
    end
end
