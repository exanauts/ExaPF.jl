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
import ExaPF: PowerSystem

const PS = PowerSystem

@testset "Compute reduced gradient on CPU" begin
    @testset "Case $case" for case in ["case9.m"] #, "case30.m"]
        datafile = joinpath(dirname(@__FILE__), "..", "data", case)
        tolerance = 1e-8
        pf = PS.PowerNetwork(datafile)
        pv = pf.pv ; npv = length(pv)
        pq = pf.pq ; npq = length(pq)
        ref = pf.ref ; nref = length(ref)
        nbus = pf.nbus

        polar = PolarForm(pf, CPU())
        cache = ExaPF.get(polar, ExaPF.PhysicalState())
        ExaPF.init_buffer!(polar, cache)

        xk = ExaPF.initial(polar, State())
        u = ExaPF.initial(polar, Control())

        jx, ju, ∂obj = ExaPF.init_autodiff_factory(polar, cache)

        # solve power flow
        conv = powerflow(polar, jx, cache, tol=1e-12)
        # No need to recompute ∇gₓ
        ∇gₓ = jx.J
        ∇gᵤ = ExaPF.jacobian(polar, ju, cache, AutoDiff.ControlJacobian())
        # test jacobian wrt x
        ∇gᵥ = ExaPF.jacobian(polar, jx, cache, AutoDiff.StateJacobian())
        @test isapprox(∇gₓ, ∇gᵥ)

        vm = cache.vmag
        va = cache.vang
        pg = cache.pg
        # Test with Matpower's Jacobian
        V = vm .* exp.(im * va)
        Ybus = pf.Ybus
        J = ExaPF.residual_jacobian(State(), V, Ybus, pf.pv, pf.pq, pf.ref)
        @test isapprox(∇gₓ, J)
        # Hessian vector product
        l = rand(length(xk))
        H = ExaPF.residual_hessian(State(), State(), V, Ybus, l, pv, pq, ref)

        function jac_diff(x)
            vm_ = copy(vm)
            va_ = copy(va)
            va_[pv] = x[1:npv]
            va_[pq] = x[npv+1:npv+npq]
            vm_[pq] = x[npv+npq+1:end]
            V = vm_ .* exp.(im * va_)
            Jx = ExaPF.residual_jacobian(State(), V, Ybus, pf.pv, pf.pq, pf.ref)
            vec = l
            return Jx' * vec
        end

        x = [va[pv] ; va[pq] ; vm[pq]]
        H_fd = FiniteDiff.finite_difference_jacobian(jac_diff, x)
        @test isapprox(H_fd, H_fd', atol=1e-5)
        @test isapprox(H, H_fd, atol=1e-5)

        J = ExaPF.residual_jacobian(Control(), V, Ybus, pv, pq, ref)
        function jac_u_diff(u)
            vm_ = copy(vm)
            va_ = copy(va)
            vm_[ref] = u[1:nref]
            vm_[pv] = u[nref+1:end]
            V = vm_ .* exp.(im * va_)
            Ju = ExaPF.residual_jacobian(Control(), V, Ybus, pv, pq, ref)[:, [ref; pv]]
            vec = l
            return Ju' * vec
        end

        u = [vm[ref]; vm[pv]]
        Hᵤᵤ_fd = FiniteDiff.finite_difference_jacobian(jac_u_diff, u)
        Hᵤᵤ = ExaPF.residual_hessian(Control(), Control(), V, Ybus, l, pv, pq, ref)

        @test isapprox(Hᵤᵤ, Hᵤᵤ_fd, atol=1e-5)

        function jac_xu_diff(x)
            vm_ = copy(vm)
            va_ = copy(va)
            va_[pv] = x[1:npv]
            va_[pq] = x[npv+1:npv+npq]
            vm_[pq] = x[npv+npq+1:end]
            V = vm_ .* exp.(im * va_)
            Jx = ExaPF.residual_jacobian(Control(), V, Ybus, pv, pq, ref)[:, [ref; pv]]
            vec = l
            return Jx' * vec
        end

        x = [va[pv] ; va[pq] ; vm[pq]]
        Hₓᵤ_fd = FiniteDiff.finite_difference_jacobian(jac_xu_diff, x)
        Hₓᵤ = ExaPF.residual_hessian(State(), Control(), V, Ybus, l, pv, pq, ref)

        @test isapprox(Hₓᵤ_fd, Hₓᵤ, atol=1e-5)
    end
end
