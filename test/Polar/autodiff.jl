
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

        jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
        ju = AutoDiff.Jacobian(polar, ExaPF.power_balance, Control())

        # solve power flow
        conv = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-12))
        V = cache.vmag .* exp.(im .* cache.vang)

        # Test Jacobian w.r.t. State
        @testset "Constraint $(cons)" for cons in [
            ExaPF.voltage_magnitude_constraints,
            ExaPF.power_balance,
            ExaPF.active_power_constraints,
            ExaPF.reactive_power_constraints,
            ExaPF.flow_constraints,
        ]
            m = ExaPF.size_constraint(polar, cons)
            # Allocation
            pbm = AutoDiff.TapeMemory(polar, cons, typeof(u))
            tgt = rand(m)
            c = zeros(m)

            ## STATE JACOBIAN

            xjacobianAD = ExaPF.AutoDiff.Jacobian(polar, cons, State())
            # Evaluate Jacobian with AD
            J = AutoDiff.jacobian!(polar, xjacobianAD, cache)
            # Matpower Jacobian
            Jmat_x = ExaPF.matpower_jacobian(polar, State(), cons, V)
            # Evaluate Jacobian transpose vector product

            # Compare with FiniteDiff
            function jac_fd_x(x)
                cache.vang[pv] .= x[1:npv]
                cache.vang[pq] .= x[npv+1:npv+npq]
                cache.vmag[pq] .= x[npv+npq+1:end]
                c = zeros(m)
                cons(polar, c, cache.vmag, cache.vang, cache.pinj, cache.qinj)
                return c
            end
            x = [cache.vang[pv]; cache.vang[pq]; cache.vmag[pq]]
            Jd = FiniteDiff.finite_difference_jacobian(jac_fd_x, x)
            @test size(J) == (m, length(x))
            # Test that AutoDiff matches MATPOWER
            @test isapprox(Jd, xjacobianAD.J, rtol=1e-5)
            @test isapprox(Jmat_x, xjacobianAD.J, rtol=1e-4)

            ## JACOBIAN VECTOR PRODUCT
            ExaPF.jtprod!(polar, pbm, cache, tgt)
            ∂cons = pbm.stack
            @test isapprox(∂cons.∂x, xjacobianAD.J' * tgt, rtol=1e-6)

            ## CONTROL JACOBIAN
            if !isa(cons, typeof(ExaPF.voltage_magnitude_constraints))
                ujacobianAD = ExaPF.AutoDiff.Jacobian(polar, cons, Control())
                # Evaluate Jacobian with AD
                J = AutoDiff.jacobian!(polar, ujacobianAD, cache)
                # Matpower Jacobian
                Jmat_u = ExaPF.matpower_jacobian(polar, Control(), cons, V)

                # Compare with FiniteDiff
                function jac_fd_u(u)
                    cache.vmag[ref] .= u[1:nref]
                    cache.vmag[pv] .= u[nref+1:npv+nref]
                    cache.pinj[pv] .= u[nref+npv+1:end]
                    c = zeros(m)
                    cons(polar, c, cache.vmag, cache.vang, cache.pinj, cache.qinj)
                    return c
                end
                u = [cache.vmag[ref]; cache.vmag[pv]; cache.pinj[pv]]
                Jd = FiniteDiff.finite_difference_jacobian(jac_fd_u, u)
                @test size(J) == (m, length(u))
                @test isapprox(Jd, ujacobianAD.J, rtol=1e-5)
                @test isapprox(Jmat_u, ujacobianAD.J, rtol=1e-6)
                @test isapprox(∂cons.∂u, ujacobianAD.J' * tgt, rtol=1e-6)
            end

            # ADJOINT
            ExaPF.adjoint!(polar, pbm, tgt, c, cache)
            function test_fd(vvm)
                cache.vmag .= vvm[1:nbus]
                cache.vang .= vvm[1+nbus:2*nbus]
                cons(polar, c, cache.vmag, cache.vang, cache.pinj, cache.qinj)
                return dot(c, tgt)
            end
            vv = [cache.vmag; cache.vang]
            vv_fd = FiniteDiff.finite_difference_gradient(test_fd, vv)
            @test isapprox(vv_fd[1:nbus], ∂cons.∂vm, rtol=1e-6)
            @test isapprox(vv_fd[1+nbus:2*nbus], ∂cons.∂va, rtol=1e-6)
        end
    end
end
