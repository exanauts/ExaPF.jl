
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

        # w.r.t. State
        for cons in [
            ExaPF.power_balance,
            ExaPF.reactive_power_constraints,
        ]
            m = ExaPF.size_constraint(polar, cons)
            mappv = [i + nbus for i in pv]
            mappq = [i + nbus for i in pq]
            # Ordering for x is (θ_pv, θ_pq, v_pq)
            statemap = vcat(mappv, mappq, pq)

            # Then, create a Jacobian object
            xjacobianAD = ExaPF.AutoDiff.Jacobian(cons, polar, statemap, State())

            # Evaluate Jacobian with AD
            J = xjacobianAD(polar, cache)

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
            @test isapprox(Jd, xjacobianAD.J, rtol=1e-6)
        end

        # w.r.t. Control
        for cons in [
            ExaPF.power_balance,
            ExaPF.reactive_power_constraints,
        ]
            m = ExaPF.size_constraint(polar, cons)
            mapu = polar.controljacobianstructure.map
            # Then, create a Jacobian object
            ujacobianAD = ExaPF.AutoDiff.Jacobian(cons, polar, mapu, Control())

            # Evaluate Jacobian with AD
            J = ujacobianAD(polar, cache)

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
            @test isapprox(Jd, ujacobianAD.J, rtol=1e-6)
        end

    end
end
