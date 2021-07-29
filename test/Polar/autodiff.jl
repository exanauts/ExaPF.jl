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

    println(devnull, jx)

    # Solve power flow
    conv = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-12))
    # Get solution in complex form.
    V = ExaPF.voltage_host(cache)

    # Test Jacobian w.r.t. State
    @testset "Constraint $(cons)" for cons in [
        ExaPF.voltage_magnitude_constraints,
        ExaPF.power_balance,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
        ExaPF.flow_constraints,
        ExaPF.bus_power_injection,
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
        # Matpower Jacobian
        Jmat_u = ExaPF.matpower_jacobian(polar, Control(), cons, V)
        Jacobian = ExaPF.is_linear(polar, cons) ? ExaPF.AutoDiff.ConstantJacobian : ExaPF.AutoDiff.Jacobian
        ujacobianAD = Jacobian(polar, cons, Control())
        # Evaluate Jacobian with AD
        J = AutoDiff.jacobian!(polar, ujacobianAD, cache)

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
        if !isnothing(ujacobianAD.J)
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

    # Test Jacobian w.r.t. State
    @testset "Constraint $(cons)" for cons in [
        ExaPF.voltage_magnitude_constraints,
        ExaPF.power_balance,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
        ExaPF.flow_constraints,
        ExaPF.bus_power_injection,
        ExaPF.network_operations,
    ]
        m = ExaPF.size_constraint(polar, cons)
        pbm = AutoDiff.TapeMemory(polar, cons, typeof(u))
        tgt = rand(m) |> MT
        c = zeros(m) |> MT
        # ADJOINT
        cons(polar, c, cache)
        ExaPF.adjoint!(polar, pbm, tgt, c, cache)
        function test_fd(vvm)
            cache.vmag .= vvm[1:nbus]
            cache.vang .= vvm[1+nbus:2*nbus]
            cons(polar, c, cache.vmag, cache.vang, cache.pnet, cache.qnet, cache.pload, cache.qload)
            return dot(c, tgt)
        end
        vv = [cache.vmag; cache.vang]
        vv_fd = FiniteDiff.finite_difference_jacobian(test_fd, vv)
        # Loosen the tolerance to 1e-5 there (finite_difference_jacobian
        # is less accurate than finite_difference_gradient)
        @test myisapprox(vv_fd[1:nbus], pbm.stack.∂vm, rtol=1e-5)
        @test myisapprox(vv_fd[1+nbus:2*nbus], pbm.stack.∂va, rtol=1e-5)
    end
end

function test_full_space_jacobian(polar, device, MT)
    pf = polar.network
    pv = pf.pv ; npv = length(pv)
    pq = pf.pq ; npq = length(pq)
    ref = pf.ref ; nref = length(ref)
    constraints = [
        ExaPF.voltage_magnitude_constraints,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
        ExaPF.flow_constraints,
    ]

    m = sum(ExaPF.size_constraint.(Ref(polar), constraints))

    buffer = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, buffer)
    # Init Jacobian storage
    jac = ExaPF.ConstraintsJacobianStorage(polar, constraints)
    # Update State and Control Jacobians
    ExaPF.update_full_jacobian!(polar, jac, buffer)
    Jx = jac.Jx |> SparseMatrixCSC |> Array
    Ju = jac.Ju |> SparseMatrixCSC |> Array

    function jac_fd_x(x)
        buffer.vang[pv] .= x[1:npv]
        buffer.vang[pq] .= x[npv+1:npv+npq]
        buffer.vmag[pq] .= x[npv+npq+1:end]
        g = similar(x, m)
        f, t = 1, 0
        for cons in constraints
            m_ = ExaPF.size_constraint(polar, cons)
            t += m_
            g_ = @view g[f:t]
            cons(polar, g_, buffer.vmag, buffer.vang, buffer.pnet, buffer.qnet, buffer.pload, buffer.qload)
            f += m_
        end
        return g
    end
    x = [buffer.vang[pv]; buffer.vang[pq]; buffer.vmag[pq]]
    Jd = FiniteDiff.finite_difference_jacobian(jac_fd_x, x) |> Array
    @test isapprox(Jd, Jx, rtol=1e-5)

    function jac_fd_u(u)
        buffer.vmag[ref] .= u[1:nref]
        buffer.vmag[pv] .= u[nref+1:npv+nref]
        buffer.pnet[pv] .= u[nref+npv+1:end]
        g = similar(x, m)
        f, t = 1, 0
        for cons in constraints
            m_ = ExaPF.size_constraint(polar, cons)
            t += m_
            g_ = @view g[f:t]
            cons(polar, g_, buffer.vmag, buffer.vang, buffer.pnet, buffer.qnet, buffer.pload, buffer.qload)
            f += m_
        end
        return g
    end
    u = [buffer.vmag[ref]; buffer.vmag[pv]; buffer.pnet[pv]]
    Jd = FiniteDiff.finite_difference_jacobian(jac_fd_u, u) |> Array
    @test isapprox(Jd, Ju, rtol=1e-5)
end

