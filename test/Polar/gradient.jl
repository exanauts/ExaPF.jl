function test_reduced_gradient(polar, device, MT)
    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)

    u = ExaPF.initial(polar, Control())
    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    ju = AutoDiff.Jacobian(polar, ExaPF.power_balance, Control())
    ∂obj = ExaPF.AdjointStackObjective(polar)
    pbm = AutoDiff.TapeMemory(ExaPF.cost_production, ∂obj, nothing)

    # Solve power flow
    conv = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-12))
    # No need to recompute ∇gₓ
    ∇gₓ = jx.J
    ∇gᵤ = AutoDiff.jacobian!(polar, ju, cache)
    # test jacobian wrt x
    ∇gᵥ = AutoDiff.jacobian!(polar, jx, cache)
    @test isequal(∇gₓ, ∇gᵥ)

    # Test with Matpower's Jacobian
    V = ExaPF.voltage_host(cache)
    Jx = ExaPF.matpower_jacobian(polar, State(), ExaPF.power_balance, V)
    Ju = ExaPF.matpower_jacobian(polar, Control(), ExaPF.power_balance, V)
    h∇gₓ = ∇gₓ |> SparseMatrixCSC |> Array
    h∇gᵤ = ∇gᵤ |> SparseMatrixCSC |> Array
    @test isapprox(h∇gₓ, Jx)
    @test isapprox(h∇gᵤ, Ju)

    ExaPF.cost_production(polar, cache)
    ExaPF.gradient_objective!(polar, pbm, cache)
    ∇fₓ = ∂obj.∇fₓ
    ∇fᵤ = ∂obj.∇fᵤ

    h∇fₓ = Array(∇fₓ)
    h∇fᵤ = Array(∇fᵤ)
    ## ADJOINT
    # lamba calculation
    λk  = -(h∇gₓ') \ h∇fₓ
    grad_adjoint = h∇fᵤ + h∇gᵤ' * λk
    # ## DIRECT
    S = - inv(h∇gₓ) * h∇gᵤ
    grad_direct = h∇fᵤ + S' * h∇fₓ
    @test isapprox(grad_adjoint, grad_direct)

    # Compare with finite difference
    function reduced_cost(u_)
        # Ensure we remain in the manifold
        ExaPF.transfer!(polar, cache, u_)
        convergence = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-14))
        return ExaPF.cost_production(polar, cache)
    end

    grad_fd = FiniteDiff.finite_difference_jacobian(reduced_cost, u)
    @test isapprox(grad_fd[:], grad_adjoint, rtol=1e-4)
end

function test_line_flow_gradient(polar, device, MT)
    u = ExaPF.initial(polar, Control())
    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)
    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    # solve power flow
    conv = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-12))

    # Adjoint of flow_constraints()
    nbus = length(cache.vmag)
    m = ExaPF.size_constraint(polar, ExaPF.flow_constraints)
    x = similar(u, 2 * nbus)
    x[1:nbus] .= cache.vmag
    x[1+nbus:2*nbus] .= cache.vang
    bus_gen = polar.indexing.index_generators
    VI = typeof(bus_gen)

    ## Example with using sum as a sort of lumping of all constraints
    function sum_constraints(x)
        VT = typeof(x)
        # Needed for ForwardDiff to have a cache with the right active type VT
        adcache = ExaPF.PolarNetworkState{VI, VT}(
            cache.vmag, cache.vang, cache.pnet, cache.qnet,
            cache.pgen, cache.qgen, cache.pload, cache.qload, cache.balance, cache.dx, bus_gen,
        )
        adcache.vmag .= x[1:nbus]
        adcache.vang .= x[1+nbus:2*nbus]
        g = VT(undef, m) ; fill!(g, 0)
        ExaPF.flow_constraints(polar, g, adcache)
        return sum(g)
    end
    # adgradg = ForwardDiff.gradient(sum_constraints,x)
    fdgradg = FiniteDiff.finite_difference_jacobian(sum_constraints,x)
    ## We pick sum() as the reduction function. This could be a mask function for active set or some log(x) for lumping.
    m_flows = ExaPF.size_constraint(polar, ExaPF.flow_constraints)
    weights = ones(m_flows) |> MT
    gradg = similar(u, 2 * nbus)
    ExaPF.flow_constraints_grad!(polar, gradg, cache, weights)
    # @test isapprox(adgradg, fdgradg)
    # TODO: The gradient is slightly off with the handwritten adjoint
    @test_broken isapprox(gradg, fdgradg[:])
end

function test_objective_adjoint(polar, device, MT)
    pf = polar.network
    nbus = pf.nbus
    pv = pf.pv ; npv = length(pv)
    pq = pf.pq ; npq = length(pq)
    ref = pf.ref ; nref = length(ref)
    pv2gen = polar.indexing.index_pv_to_gen
    nx = ExaPF.get(polar, ExaPF.NumberOfState())

    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)

    u = ExaPF.initial(polar, Control())

    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    conv = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-12))

    # Evaluate gradient
    pbm = ExaPF.pullback_objective(polar)
    ExaPF.gradient_objective!(polar, pbm, cache)

    # Compare with finite diff
    x = [cache.vang[pv] ; cache.vang[pq] ; cache.vmag[pq]]
    u = [cache.vmag[ref]; cache.vmag[pv]; cache.pgen[pv2gen]]

    function test_objective_fd(z)
        x_ = z[1:nx]
        u_ = z[1+nx:end]
        # Transfer control
        ExaPF.transfer!(polar, cache, u_)
        # Transfer state (manually)
        cache.vang[pv] .= x_[1:npv]
        cache.vang[pq] .= x_[npv+1:npv+npq]
        cache.vmag[pq] .= x_[npv+npq+1:end]
        return ExaPF.cost_production(polar, cache)
    end
    ∇f = FiniteDiff.finite_difference_jacobian(test_objective_fd, [x; u])

    @test myisapprox(∇f[1:nx], pbm.stack.∇fₓ, rtol=1e-5)
    @test myisapprox(∇f[1+nx:end], pbm.stack.∇fᵤ, rtol=1e-5)
    return
end

function test_objective_with_ramping_adjoint(polar, device, MT)
    pf = polar.network
    nbus = pf.nbus
    ngen = pf.ngen
    pv = pf.pv ; npv = length(pv)
    pq = pf.pq ; npq = length(pq)
    ref = pf.ref ; nref = length(ref)
    pv2gen = polar.indexing.index_pv_to_gen
    nx = ExaPF.get(polar, ExaPF.NumberOfState())

    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)

    u = ExaPF.initial(polar, Control())

    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    conv = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-12))
    x = [cache.vang[pv] ; cache.vang[pq] ; cache.vmag[pq]]
    u = [cache.vmag[ref]; cache.vmag[pv]; cache.pgen[pv2gen]]

    # Intermediate
    s = similar(u, ngen) ; fill!(s, 0.0)
    σ = 1.0
    ρ1 = 1.0
    ρ2 = 1.0
    τ = 1.0
    p1 = similar(u, ngen) ; fill!(p1, 0.0)
    p2 = similar(u, ngen) ; fill!(p2, 0.0)
    p3 = similar(u, ngen) ; fill!(p3, 0.0)
    λ1 = similar(u, ngen) ; fill!(λ1, 1.0)
    λ2 = similar(u, ngen) ; fill!(λ2, 1.0)
    # Evaluate gradient
    pbm = ExaPF.pullback_ramping(polar, nothing)

    for t in [0, 1, 2]
        ExaPF.adjoint_penalty_ramping_constraints!(polar, pbm, cache, s, t, σ, τ, λ1, λ2, ρ1, ρ2, p1, p2, p3)

        # Compare with finite diff
        function test_objective_fd(z)
            x_ = z[1:nx]
            u_ = z[1+nx:end]
            # Transfer control
            ExaPF.transfer!(polar, cache, u_)
            # Transfer state (manually)
            cache.vang[pv] .= x_[1:npv]
            cache.vang[pq] .= x_[npv+1:npv+npq]
            cache.vmag[pq] .= x_[npv+npq+1:end]
            return ExaPF.cost_penalty_ramping_constraints(polar, cache, s, t, σ, τ, λ1, λ2, ρ1, ρ2, p1, p2, p3)
        end
        ∇f = FiniteDiff.finite_difference_jacobian(test_objective_fd, [x; u])

        @test myisapprox(∇f[1:nx], pbm.stack.∇fₓ, rtol=1e-5)
        @test myisapprox(∇f[1+nx:end], pbm.stack.∇fᵤ, rtol=1e-5)
    end
end

