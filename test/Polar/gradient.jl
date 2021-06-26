function test_reduced_gradient(polar, device, MT)
    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)

    u = ExaPF.initial(polar, Control())
    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    ju = AutoDiff.Jacobian(polar, ExaPF.power_balance, Control())
    ∂obj = ExaPF.AdjointStackObjective(polar)
    pbm = AutoDiff.TapeMemory(ExaPF.active_power_generation, ∂obj, nothing)

    # Solve power flow
    conv = powerflow(polar, jx, cache, NewtonRaphson(tol=1e-12))
    # No need to recompute ∇gₓ
    ∇gₓ = jx.J
    ∇gᵤ = AutoDiff.jacobian!(polar, ju, cache)
    # test jacobian wrt x
    ∇gᵥ = AutoDiff.jacobian!(polar, jx, cache)
    @test isequal(∇gₓ, ∇gᵥ)

    # Test with Matpower's Jacobian
    V = cache.vmag .* exp.(im * cache.vang) |> Array
    Jx = ExaPF.matpower_jacobian(polar, State(), ExaPF.power_balance, V)
    Ju = ExaPF.matpower_jacobian(polar, Control(), ExaPF.power_balance, V)
    h∇gₓ = ∇gₓ |> SparseMatrixCSC |> Array
    h∇gᵤ = ∇gᵤ |> SparseMatrixCSC |> Array
    @test isapprox(h∇gₓ, Jx)
    @test isapprox(h∇gᵤ, Ju)
    # Refresh cache with new values of vmag and vang
    ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
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
        ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
        return ExaPF.cost_production(polar, cache)
    end

    grad_fd = FiniteDiff.finite_difference_jacobian(reduced_cost, u)
    grad_fd = grad_fd[:] |> Array # Transfer gradient to host
    @test isapprox(grad_fd, grad_adjoint, rtol=1e-4)
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
    M = typeof(u)
    m = ExaPF.size_constraint(polar, ExaPF.flow_constraints)
    x = ExaPF.xzeros(M, 2 * nbus)
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
    gradg = ExaPF.xzeros(M, 2 * nbus)
    ExaPF.flow_constraints_grad!(polar, gradg, cache, weights)
    # @test isapprox(adgradg, fdgradg)
    # TODO: The gradient is slightly off with the handwritten adjoint
    h_gradg = gradg |> Array
    h_fdgradg = fdgradg[:] |> Array
    @test_broken isapprox(gradg, fdgradg)
end

