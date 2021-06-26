function test_constraints_jacobian(polar, device, MT)
    pf = polar.network
    nbus = pf.nbus
    pv = pf.pv ; npv = length(pv)
    pq = pf.pq ; npq = length(pq)
    ref = pf.ref ; nref = length(ref)

    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)

    u = ExaPF.initial(polar, Control())

    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())

    # Solve power flow
    conv = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-12))
    # Get solution in complex form.
    V = cache.vmag .* exp.(im .* cache.vang) |> Array

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
        tgt = rand(m) |> MT
        c = zeros(m) |> MT

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
            c = zeros(m) |> MT
            cons(polar, c, cache.vmag, cache.vang, cache.pnet, cache.qnet, cache.pload, cache.qload)
            return c
        end
        x = [cache.vang[pv]; cache.vang[pq]; cache.vmag[pq]]
        Jd = FiniteDiff.finite_difference_jacobian(jac_fd_x, x) |> Array
        Jx = xjacobianAD.J |> SparseMatrixCSC |> Array
        ## JACOBIAN VECTOR PRODUCT
        ExaPF.jacobian_transpose_product!(polar, pbm, cache, tgt)
        ∂cons = pbm.stack

        @test size(J) == (m, length(x))
        @test isapprox(Jd, Jx, rtol=1e-5)
        @test isapprox(Jmat_x, Jx, rtol=1e-4)
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
                cache.pnet[pv] .= u[nref+npv+1:end]
                c = zeros(m) |> MT
                cons(polar, c, cache.vmag, cache.vang, cache.pnet, cache.qnet, cache.pload, cache.qload)
                return c
            end
            u = [cache.vmag[ref]; cache.vmag[pv]; cache.pnet[pv]]
            Jd = FiniteDiff.finite_difference_jacobian(jac_fd_u, u) |> Array
            Ju = ujacobianAD.J |> SparseMatrixCSC |> Array
            @test size(J) == (m, length(u))
            @test isapprox(Jd, Ju, rtol=1e-5)
            @test isapprox(Jmat_u, Ju, rtol=1e-6)
            @test isapprox(∂cons.∂u, ujacobianAD.J' * tgt, rtol=1e-6)
        end
    end
end

function test_constraints_adjoint(polar, device, MT)
    pf = polar.network
    nbus = pf.nbus
    pv = pf.pv ; npv = length(pv)
    pq = pf.pq ; npq = length(pq)
    ref = pf.ref ; nref = length(ref)

    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)

    u = ExaPF.initial(polar, Control())

    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    conv = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-12))
    # Get solution in complex form.
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
        pbm = AutoDiff.TapeMemory(polar, cons, typeof(u))
        tgt = rand(m) |> MT
        c = zeros(m) |> MT
        # ADJOINT
        ExaPF.adjoint!(polar, pbm, tgt, c, cache)
        function test_fd(vvm)
            cache.vmag .= vvm[1:nbus]
            cache.vang .= vvm[1+nbus:2*nbus]
            cons(polar, c, cache.vmag, cache.vang, cache.pnet, cache.qnet, cache.pload, cache.qload)
            return dot(c, tgt)
        end
        vv = [cache.vmag; cache.vang]
        vv_fd = FiniteDiff.finite_difference_jacobian(test_fd, vv)
        # Transfer data back to the CPU
        h_vm = pbm.stack.∂vm |> Array
        h_va = pbm.stack.∂va |> Array
        h_vv_fd = vv_fd |> Array
        @test isapprox(h_vv_fd[1:nbus], h_vm, rtol=1e-6)
        @test isapprox(h_vv_fd[1+nbus:2*nbus], h_va, rtol=1e-6)
    end
end

