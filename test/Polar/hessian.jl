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
        ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
        @test conv.has_converged
        # No need to recompute ∇gₓ
        ∇gₓ = jx.J
        ∇gᵤ = AutoDiff.jacobian!(polar, ju, cache)
        # test jacobian wrt x
        ∇gᵥ = AutoDiff.jacobian!(polar, jx, cache)
        @test isequal(∇gₓ, ∇gᵥ)

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
        Jₓ = ExaPF.matpower_jacobian(polar, State(), ExaPF.power_balance, V)
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
            Jx = ExaPF.matpower_jacobian(polar, State(), ExaPF.power_balance, V)
            return Jx' * λ
        end

        # Evaluate Hessian-vector product (full ∇²gₓₓ is a 3rd dimension tensor)
        ∇²gλ = ExaPF.residual_hessian(V, Ybus, λ, pv, pq, ref)
        H_fd = FiniteDiff.finite_difference_jacobian(jac_diff, x)
        @test isapprox(∇²gλ.xx, H_fd, rtol=1e-6)
        # Hessian-vector product using forward over adjoint AD
        ybus_re, ybus_im = ExaPF.Spmat{Vector{Int}, Vector{Float64}}(Ybus)
        pbus = real(pf.sbus)
        qbus = imag(pf.sbus)
        F = zeros(Float64, npv + 2*npq)
        nx = size(∇²gλ.xx,1)
        nu = size(∇²gλ.uu,1)
        HessianAD = ExaPF.AutoDiff.Hessian(polar.hessianstructure, F, vm, va,
                                                    ybus_re, ybus_im, pbus, qbus, pf.pv, pf.pq, pf.ref)
        tgt = rand(nx + nu)
        tgt .= 1.0
        # set tangets only for x direction
        tgt[nx+1:end] .= 0.0
        projxx = ExaPF.AutoDiff.tgt_adj_residual_hessian!(
            HessianAD, ExaPF.adj_residual_polar!, λ, tgt, vm, va,
            ybus_re, ybus_im, pbus, qbus, pf.pv, pf.pq, pf.ref, nbus)
        @test isapprox(projxx[1:nx], ∇²gλ.xx * tgt[1:nx])
        tgt = rand(nx + nu)
        # set tangets only for u direction
        tgt[1:nx] .= 0.0
        projuu = ExaPF.AutoDiff.tgt_adj_residual_hessian!(
            HessianAD, ExaPF.adj_residual_polar!, λ, tgt, vm, va,
            ybus_re, ybus_im, pbus, qbus, pf.pv, pf.pq, pf.ref, nbus)
        @test isapprox(projuu[nx+1:end], ∇²gλ.uu * tgt[nx+1:end])
        # check cross terms ux
        tgt = rand(nx + nu)
        # Build full Hessian
        H = [
            ∇²gλ.xx ∇²gλ.xu';
            ∇²gλ.xu ∇²gλ.uu
        ]
        projxu = ExaPF.AutoDiff.tgt_adj_residual_hessian!(
            HessianAD, ExaPF.adj_residual_polar!, λ, tgt, vm, va,
            ybus_re, ybus_im, pbus, qbus, pf.pv, pf.pq, pf.ref, nbus)
        @test isapprox(projxu, H * tgt)

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

        if !iszero(∇²gλ.uu[1:nref+npv, 1:nref+npv])
            @test isapprox(∇²gλ.uu[1:nref+npv, 1:nref+npv], Hᵤᵤ_fd[1:nref+npv, :], rtol=1e-6)
        end

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
        @test isapprox(∇²gλ.xu[1:nref+npv, :], Hₓᵤ_fd, rtol=1e-6)

        ##################################################
        # Step 3: computation of Hessian of objective f
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
            ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
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
        ∇²f = ExaPF.hessian_cost(polar, cache)
        ∇²fₓₓ = ∇²f.xx
        ∇²fᵤᵤ = ∇²f.uu
        ∇²fₓᵤ = ∇²f.xu
        @test isapprox(∇²fₓₓ, H_ffd[1:nx, 1:nx], rtol=1e-6)
        index_u = nx+1:nx+nref+2*npv
        @test isapprox(∇²fₓᵤ, H_ffd[index_u, 1:nx], rtol=1e-6)
        @test isapprox(∇²fᵤᵤ, H_ffd[index_u, index_u], rtol=1e-6)

        ∇gaₓ = ∇²fₓₓ + ∇²gλ.xx

        # Computation of the reduced Hessian
        function reduced_hess(w)
            # Second-order adjoint
            z = -(∇gₓ ) \ (∇gᵤ * w)
            ψ = -(∇gₓ') \ (∇²fₓᵤ' * w + ∇²gλ.xu' * w +  ∇gaₓ * z)
            Hw = ∇²fᵤᵤ * w +  ∇²gλ.uu * w + ∇gᵤ' * ψ  + ∇²fₓᵤ * z + ∇²gλ.xu * z
            return Hw
        end

        w = zeros(nu)
        H = zeros(nu, nu)
        for i in 1:nu
            fill!(w, 0)
            w[i] = 1.0
            H[:, i] .= reduced_hess(w)
        end

        ##################################################
        # Step 4: include constraints in Hessian
        ##################################################
        # h1 (state)      : xl <= x <= xu
        # h2 (by-product) : yl <= y <= yu
        # Test sequential evaluation of Hessian
        local ∂₂Q
        for cons in [ExaPF.voltage_magnitude_constraints, ExaPF.active_power_constraints, ExaPF.reactive_power_constraints]
            m = ExaPF.size_constraint(polar, cons)
            λq = ones(m)
            ∂₂Q = ExaPF.hessian(polar, cons, cache, λq)
        end

        μ = rand(ngen)
        ∂₂Q = ExaPF.hessian(polar, ExaPF.reactive_power_constraints, cache, μ)
        function jac_x(z)
            x_ = z[1:nx]
            u_ = z[1+nx:end]
            # Transfer control
            ExaPF.transfer!(polar, cache, u_)
            # Transfer state (manually)
            cache.vang[pv] .= x_[1:npv]
            cache.vang[pq] .= x_[npv+1:npv+npq]
            cache.vmag[pq] .= x_[npv+npq+1:end]
            ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
            J = ExaPF.jacobian(polar, ExaPF.reactive_power_constraints, cache)
            return [J.x J.u]' * μ
        end

        H_fd = FiniteDiff.finite_difference_jacobian(jac_x, [x; u])
        @test isapprox(∂₂Q.uu, H_fd[nx+1:end, nx+1:end], rtol=1e-6)
        @test isapprox(∂₂Q.xx, H_fd[1:nx, 1:nx], rtol=1e-6)
        @test isapprox(∂₂Q.xu, H_fd[nx+1:end, 1:nx], rtol=1e-6)
    end
end

