
@testset "Compute Autodiff on CPU" begin
    @testset "Case $case" for case in ["case9.m", "case30.m"]
        datafile = joinpath(dirname(@__FILE__), "..", "..", "data", case)
        tolerance = 1e-8
        pf = PS.PowerNetwork(datafile)
        nbus = pf.nbus
        pv = pf.pv ; npv = length(pv)
        pq = pf.pq ; npq = length(pq)
        ref = pf.ref ; nref = length(ref)

        polar = PolarForm(pf, CPU())
        cache = ExaPF.get(polar, ExaPF.PhysicalState())
        ExaPF.init_buffer!(polar, cache)

        xk = ExaPF.initial(polar, State())
        u = ExaPF.initial(polar, Control())

        jx, ju, ∂obj = ExaPF.init_autodiff_factory(polar, cache)

        # solve power flow
        conv = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-12))
        V = cache.vmag .* exp.(im .* cache.vang)

        # Test Jacobian w.r.t. State
        for cons in [
            ExaPF.voltage_magnitude_constraints,
            ExaPF.power_balance,
            ExaPF.reactive_power_constraints,
            ExaPF.flow_constraints,
        ]
            m = ExaPF.size_constraint(polar, cons)
            xjacobianAD = ExaPF.AutoDiff.Jacobian(polar, cons, State())
            # Evaluate Jacobian with AD
            J = AutoDiff.jacobian!(polar, xjacobianAD, cache)
            # Matpower Jacobian
            Jmat = ExaPF.matpower_jacobian(polar, State(), cons, V)
            # Evaluate Jacobian transpose vector product
            tgt = rand(m)
            ExaPF.jtprod(polar, cons, ∂obj, cache, tgt)

            # Compare with FiniteDiff
            function jac_fd_x(x)
                cache.vang[pv] .= x[1:npv]
                cache.vang[pq] .= x[npv+1:npv+npq]
                cache.vmag[pq] .= x[npv+npq+1:end]
                c = zeros(m)
                cons(polar, c, cache)
                return c
            end
            x = [cache.vang[pv]; cache.vang[pq]; cache.vmag[pq]]
            Jd = FiniteDiff.finite_difference_jacobian(jac_fd_x, x)
            @test isapprox(Jd, xjacobianAD.J, rtol=1e-5)
            @test isapprox(Jmat, xjacobianAD.J, rtol=1e-5)
            @test isapprox(∂obj.∇fₓ, xjacobianAD.J' * tgt, rtol=1e-6)
        end

        # Test Jacobian w.r.t. Control
        for cons in [
            # ExaPF.voltage_magnitude_constraints, # TODO: handle case where Jacobian is zero
            ExaPF.power_balance,
            ExaPF.reactive_power_constraints,
            ExaPF.flow_constraints,
        ]
            m = ExaPF.size_constraint(polar, cons)
            ujacobianAD = ExaPF.AutoDiff.Jacobian(polar, cons, Control())
            # Evaluate Jacobian with AD
            J = AutoDiff.jacobian!(polar, ujacobianAD, cache)
            # Matpower Jacobian
            Jmat = ExaPF.matpower_jacobian(polar, Control(), cons, V)
            # Evaluate Jacobian transpose vector product
            adj = rand(m)
            ExaPF.jtprod(polar, cons, ∂obj, cache, adj)

            # Compare with FiniteDiff
            function jac_fd_u(u)
                cache.vmag[ref] .= u[1:nref]
                cache.vmag[pv] .= u[nref+1:npv+nref]
                cache.pinj[pv] .= u[nref+npv+1:end]
                c = zeros(m)
                cons(polar, c, cache)
                return c
            end
            u = [cache.vmag[ref]; cache.vmag[pv]; cache.pinj[pv]]
            Jd = FiniteDiff.finite_difference_jacobian(jac_fd_u, u)
            @test isapprox(Jd, ujacobianAD.J, rtol=1e-5)
            @test isapprox(Jmat, ujacobianAD.J, rtol=1e-6)
            @test isapprox(∂obj.∇fᵤ, ujacobianAD.J' * adj, rtol=1e-6)
        end

        # Test adjoint
        for cons in [
            ExaPF.voltage_magnitude_constraints,
            ExaPF.power_balance,
            ExaPF.reactive_power_constraints,
            ExaPF.flow_constraints,
        ]
            m = ExaPF.size_constraint(polar, cons)
            λ = rand(m)
            c = zeros(m)
            ExaPF.adjoint!(polar, cons, λ, c, ∂obj, cache)
            function test_fd(vvm)
                cache.vmag .= vvm[1:nbus]
                cache.vang .= vvm[1+nbus:2*nbus]
                cons(polar, c, cache)
                return dot(c, λ)
            end
            vv = [cache.vmag; cache.vang]
            vv_fd = FiniteDiff.finite_difference_gradient(test_fd, vv)
            @test isapprox(vv_fd[1:nbus], ∂obj.∂vm, rtol=1e-6)
            @test isapprox(vv_fd[1+nbus:2*nbus], ∂obj.∂va, rtol=1e-6)
        end
    end
end
