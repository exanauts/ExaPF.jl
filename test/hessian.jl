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
        datafile = joinpath(dirname(@__FILE__), "..", "data", case)
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
            Ju = ExaPF.residual_jacobian(Control(), V, Ybus, pv, pq, ref)[:, 1:nref+npv]
            return Ju' * λ
        end

        Hᵤᵤ_fd = FiniteDiff.finite_difference_jacobian(jac_u_diff, u)
        ∇²gᵤᵤλ = ExaPF.residual_hessian(Control(), Control(), V, Ybus, λ, pv, pq, ref)

        @test isapprox(∇²gᵤᵤλ, Hᵤᵤ_fd, atol=1e-3)

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
        @test isapprox(∇²gₓᵤλ, Hₓᵤ_fd, rtol=1e-3)

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

        # Hessian of active power generation at slack node (this is a matrix)
        ∂₂Pₓₓ = ExaPF.active_power_hessian(State(), State(), V, Ybus, pv, pq, ref)
        ∂₂Pₓᵤ = ExaPF.active_power_hessian(State(), Control(), V, Ybus, pv, pq, ref)
        ∂₂Pᵤᵤ = ExaPF.active_power_hessian(Control(), Control(), V, Ybus, pv, pq, ref)
        @test issymmetric(∂₂Pₓₓ)
        @test issymmetric(∂₂Pᵤᵤ)

        Jpgₓ = ExaPF.active_power_jacobian(State(), V, Ybus, pv, pq, ref)
        Jpgᵤ = ExaPF.active_power_jacobian(Control(), V, Ybus, pv, pq, ref)
        ∂Pₓ = Jpgₓ[ref, :]
        ∂Pᵤ = Jpgᵤ[ref, :]  # take only Jacobian w.r.t. v_ref and v_pv

        # Hessians of objective
        ∇²fₓₓ = ∂c .* ∂₂Pₓₓ + ∂²c .* ∂Pₓ' * ∂Pₓ
        ∇²fᵤᵤ = ∂c .* ∂₂Pᵤᵤ + ∂²c .* ∂Pᵤ' * ∂Pᵤ
        ∇²fₓᵤ = ∂c .* ∂₂Pₓᵤ + ∂²c .* ∂Pᵤ' * ∂Pₓ
        # TODO: test broke on case30 because of indexing issues
        @test isapprox(∇²fₓₓ, H_ffd[1:nx, 1:nx], rtol=1e-3)
        index_xu = nx+1:nx+nref+npv
        @test isapprox(∇²fₓᵤ, H_ffd[index_xu, 1:nx], rtol=1e-3)

        ∇gaₓ = ∇²fₓₓ + ∇²gₓₓλ

        # Computation of the reduced Hessian
        function reduced_hess(w)
            wᵥ = w[1:nref+npv]
            wₚ = w[nref+npv+1:nref+2*npv]
            # Second-order adjoint
            z = -(∇gₓ ) \ (∇gᵤ * w)
            ψ = -(∇gₓ') \ (∇²fₓᵤ' * wᵥ + ∇²gₓᵤλ' * wᵥ +  ∇gaₓ * z)
            ∇gᵤᵛ = ∇gᵤ[: , 1:nref+npv]
            ∇gᵤᵖ = ∇gᵤ[: , nref+npv+1:nref+2*npv]
            # Hessian w.r.t. voltages
            Hwᵥ = ∇²fᵤᵤ * wᵥ +  ∇²gᵤᵤλ * wᵥ + ∇gᵤᵛ' * ψ  + ∇²fₓᵤ * z + ∇²gₓᵤλ * z
            # Hessian w.r.t. active power generation
            Hwₚ = ∂²cpv .* wₚ + ∇gᵤᵖ' * ψ
            Hw = [Hwᵥ; Hwₚ]
            return Hw
        end

        w = zeros(nu)
        H = zeros(nu, nu)
        for i in 1:nu
            fill!(w, 0)
            w[i] = 1.0
            H[:, i] .= reduced_hess(w)
        end
        @info("h", H)

        ##################################################
        # Step 3: include constraints in Hessian
        ##################################################
        # h1 (state)      : xl <= x <= xu
        # h2 (by-product) : yl <= y <= yu
        λq = ones(ngen)

        ∂₂Qₓₓ = ExaPF.reactive_power_hessian(State(), State(), V, Ybus, λq, pv, pq, ref, gen2bus)
        ∂₂Qₓᵤ = ExaPF.reactive_power_hessian(State(), Control(), V, Ybus, λq, pv, pq, ref, gen2bus)
        ∂₂Qᵤᵤ = ExaPF.reactive_power_hessian(Control(), Control(), V, Ybus, λq, pv, pq, ref, gen2bus)
        Jh1ₓ, Jh1ᵤ = ExaPF.jacobian(polar, ExaPF.state_constraint, cache)
        Jh2ₓ, Jh2ᵤ = ExaPF.jacobian(polar, ExaPF.power_constraints, cache)
        Jₓ = [Jh1ₓ; Jh2ₓ]
        Jᵤ = [Jh1ᵤ; Jh2ᵤ]
        Jᵤᵛ = Jᵤ[: , 1:nref+npv]
        Jᵤᵖ = Jᵤ[: , nref+npv+1:nref+2*npv]

        # Hessians of objective: add penalty terms
        ∇²fₓₓ = ∇²fₓₓ + ∂₂Qₓₓ + Jₓ' * Jₓ
        ∇²fᵤᵤ = ∇²fᵤᵤ + ∂₂Qᵤᵤ + Jᵤᵛ' * Jᵤᵛ
        ∇²fₓᵤ = ∇²fₓᵤ + ∂₂Qₓᵤ + Jᵤᵛ' * Jₓ

        ∇gaₓ = ∇²fₓₓ + ∇²gₓₓλ

        w = zeros(nu)
        H = zeros(nu, nu)
        for i in 1:nu
            fill!(w, 0)
            w[i] = 1.0
            H[:, i] .= reduced_hess(w)
        end
    end
end

