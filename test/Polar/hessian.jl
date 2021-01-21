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

# Warning: currently works only on CPU, as depends on
# explicit evaluation of Hessian, using MATPOWER expressions
@testset "Compute reduced gradient on CPU" begin
    @testset "Case $case" for case in ["case9.m", "case30.m"]
        ##################################################
        # Initialization
        ##################################################
        datafile = joinpath(dirname(@__FILE__), "..", "..", "data", case)
        tolerance = 1e-8
        pf = PS.PowerNetwork(datafile)
        polar = PolarForm(pf, CPU())
        pv = pf.pv ; npv = length(pv)
        pq = pf.pq ; npq = length(pq)
        ref = pf.ref ; nref = length(ref)
        nbus = pf.nbus
        ngen = get(polar, PS.NumberOfGenerators())

        pv2gen = polar.indexing.index_pv_to_gen
        ref2gen = polar.indexing.index_ref_to_gen
        gen2bus = polar.indexing.index_generators
        cache = ExaPF.get(polar, ExaPF.PhysicalState())
        ExaPF.init_buffer!(polar, cache)

        xk = ExaPF.initial(polar, State())
        u = ExaPF.initial(polar, Control())
        nx = length(xk) ; nu = length(u)

        jx, ju, ∂obj = ExaPF.init_autodiff_factory(polar, cache)

        ##################################################
        # Step 1: computation of first-order adjoint
        ##################################################
        conv = powerflow(polar, jx, cache, NewtonRaphson())
        ExaPF.update!(polar, PS.Generator(), PS.ActivePower(), cache)
        @test conv.has_converged
        # No need to recompute ∇gₓ
        ∇gₓ = jx.J
        ∇gᵤ = ExaPF.jacobian(polar, ju, cache, AutoDiff.ControlJacobian())
        # test jacobian wrt x
        ∇gᵥ = ExaPF.jacobian(polar, jx, cache, AutoDiff.StateJacobian())
        @test isapprox(∇gₓ, ∇gᵥ)

        # Fetch values found by Newton-Raphson algorithm
        vm = cache.vmag
        va = cache.vang
        pg = cache.pg
        # State & Control
        x = [va[pv] ; va[pq] ; vm[pq]]
        u = [vm[ref]; vm[pv]]
        # Test with Matpower's Jacobian
        V = vm .* exp.(im * va)
        Ybus = pf.Ybus
        Jₓ = ExaPF.residual_jacobian(State(), V, Ybus, pf.pv, pf.pq, pf.ref)
        @test isapprox(∇gₓ, Jₓ )
        # Hessian vector product
        ExaPF.∂cost(polar, ∂obj, cache)
        ∇fₓ = ∂obj.∇fₓ
        ∇fᵤ = ∂obj.∇fᵤ
        λ  = -(∇gₓ') \ ∇fₓ
        grad_adjoint = ∇fᵤ + ∇gᵤ' * λ

        ##################################################
        # Step 2: computation of Hessian of powerflow g
        ##################################################
        ## w.r.t. xx
        function jac_diff(x)
            vm_ = copy(vm)
            va_ = copy(va)
            va_[pv] = x[1:npv]
            va_[pq] = x[npv+1:npv+npq]
            vm_[pq] = x[npv+npq+1:end]
            V = vm_ .* exp.(im * va_)
            Jx = ExaPF.residual_jacobian(State(), V, Ybus, pf.pv, pf.pq, pf.ref)
            return Jx' * λ
        end

        # Evaluate Hessian-vector product (full ∇²gₓₓ is a 3rd dimension tensor)
        ∇²gₓₓλ = ExaPF.residual_hessian(State(), State(), V, Ybus, λ, pv, pq, ref)
        H_fd = FiniteDiff.finite_difference_jacobian(jac_diff, x)
        @test isapprox(∇²gₓₓλ, H_fd, rtol=1e-3)

        ## w.r.t. uu
        function jac_u_diff(u)
            vm_ = copy(vm)
            va_ = copy(va)
            vm_[ref] = u[1:nref]
            vm_[pv] = u[nref+1:end]
            V = vm_ .* exp.(im * va_)
            Ju = ExaPF.residual_jacobian(Control(), V, Ybus, pv, pq, ref)
            return Ju' * λ
        end

        Hᵤᵤ_fd = FiniteDiff.finite_difference_jacobian(jac_u_diff, u)
        ∇²gᵤᵤλ = ExaPF.residual_hessian(Control(), Control(), V, Ybus, λ, pv, pq, ref)

        @test isapprox(∇²gᵤᵤλ[1:nref+npv, 1:nref+npv], Hᵤᵤ_fd[1:nref+npv, :], atol=1e-3)

        ## w.r.t. xu
        function jac_xu_diff(x)
            vm_ = copy(vm)
            va_ = copy(va)
            va_[pv] = x[1:npv]
            va_[pq] = x[npv+1:npv+npq]
            vm_[pq] = x[npv+npq+1:end]
            V = vm_ .* exp.(im * va_)
            Ju = ExaPF.residual_jacobian(Control(), V, Ybus, pv, pq, ref)[:, 1:nref+npv]
            return Ju' * λ
        end

        Hₓᵤ_fd = FiniteDiff.finite_difference_jacobian(jac_xu_diff, x)
        ∇²gₓᵤλ = ExaPF.residual_hessian(State(), Control(), V, Ybus, λ, pv, pq, ref)
        @test isapprox(∇²gₓᵤλ[1:nref+npv, :], Hₓᵤ_fd, rtol=1e-3)

        ##################################################
        # Step 2: computation of Hessian of objective f
        ##################################################

        # Finite difference routine
        function cost_x(z)
            x_ = z[1:nx]
            u_ = z[1+nx:end]
            # Transfer control
            ExaPF.transfer!(polar, cache, u_)
            # Transfer state (manually)
            cache.vang[pv] .= x_[1:npv]
            cache.vang[pq] .= x_[npv+1:npv+npq]
            cache.vmag[pq] .= x_[npv+npq+1:end]
            ExaPF.update!(polar, PS.Generator(), PS.ActivePower(), cache)
            return ExaPF.cost_production(polar, cache.pg)
        end

        # Update variables
        x = [va[pv] ; va[pq] ; vm[pq]]
        u = [vm[ref]; vm[pv]; pg[pv2gen]]
        # Reset voltage
        V = vm .* exp.(im * va)
        # Adjoint
        coefs = polar.costs_coefficients
        c3 = @view coefs[:, 3]
        c4 = @view coefs[:, 4]
        # Return adjoint of quadratic cost
        adj_cost = c3 .+ 2.0 .* c4 .* pg
        # Adjoint of reference generator
        ∂c = adj_cost[ref2gen][1]
        ∂²c = 2.0 * c4[ref2gen][1]
        ∂²cpv = 2.0 * c4[pv2gen]

        H_ffd = FiniteDiff.finite_difference_hessian(cost_x, [x; u])

        # Hessians of objective
        ∇²f = ExaPF.hessian_cost(polar, ∂obj, cache)
        ∇²fₓₓ = ∇²f.xx
        ∇²fᵤᵤ = ∇²f.uu
        ∇²fₓᵤ = ∇²f.xu
        @test isapprox(∇²fₓₓ, H_ffd[1:nx, 1:nx], rtol=1e-3)
        index_xu = nx+1:nx+nref+2*npv
        @test isapprox(∇²fₓᵤ, H_ffd[index_xu, 1:nx], rtol=1e-3)

        ∇gaₓ = ∇²fₓₓ + ∇²gₓₓλ

        # Computation of the reduced Hessian
        function reduced_hess(w)
            # Second-order adjoint
            z = -(∇gₓ ) \ (∇gᵤ * w)
            ψ = -(∇gₓ') \ (∇²fₓᵤ' * w + ∇²gₓᵤλ' * w +  ∇gaₓ * z)
            Hw = ∇²fᵤᵤ * w +  ∇²gᵤᵤλ * w + ∇gᵤ' * ψ  + ∇²fₓᵤ * z + ∇²gₓᵤλ * z
            return Hw
        end

        w = zeros(nu)
        H = zeros(nu, nu)
        for i in 1:nu
            fill!(w, 0)
            w[i] = 1.0
            H[:, i] .= reduced_hess(w)
        end
        # @info("h", H)

        ##################################################
        # Step 3: include constraints in Hessian
        ##################################################
        # h1 (state)      : xl <= x <= xu
        # h2 (by-product) : yl <= y <= yu
        for cons in [ExaPF.state_constraint, ExaPF.power_constraints]
            m = ExaPF.size_constraint(polar, cons)
            λq = ones(m)
            ∂₂Q = ExaPF.hessian(polar, cons, ∂obj, cache, λq)
        end
    end
end

