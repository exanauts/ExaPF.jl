function test_polar_network_cache(polar, device, M)
    nₓ = ExaPF.get(polar, NumberOfState())
    ngen = get(polar, PS.NumberOfGenerators())
    nbus = get(polar, PS.NumberOfBuses())
    u0 = ExaPF.initial(polar, Control())

    cache = ExaPF.get(polar, ExaPF.PhysicalState())

    # By defaut, values are equal to 0
    @test iszero(cache)
    @test isa(cache.vmag, M)
    @test cache.bus_gen == polar.indexing.index_generators
    # Transfer control u0 inside cache
    ExaPF.transfer!(polar, cache, u0)
    # Test that all attributes have valid length
    @test length(cache.vang) == length(cache.vmag) == length(cache.pnet) == length(cache.qnet) == nbus
    @test length(cache.pgen) == length(cache.qgen) == length(cache.bus_gen) == ngen
    @test length(cache.dx) == length(cache.balance) == nₓ

    ## Buses
    values = similar(u0, nbus)
    fill!(values, 1)
    ExaPF.setvalues!(cache, PS.VoltageMagnitude(), values)
    ExaPF.setvalues!(cache, PS.VoltageAngle(), values)
    ExaPF.setvalues!(cache, PS.ActiveLoad(), values)
    ExaPF.setvalues!(cache, PS.ReactiveLoad(), values)
    @test cache.vmag == values
    @test cache.vang == values
    @test cache.pload == values
    @test cache.qload == values

    ## Generators
    vgens = similar(u0, ngen)
    fill!(vgens, 2.0)
    ExaPF.setvalues!(cache, PS.ActivePower(), vgens)
    ExaPF.setvalues!(cache, PS.ReactivePower(), vgens)
    genbus = polar.indexing.index_generators
    @test cache.pgen == vgens
    @test cache.qgen == vgens
    @test cache.pnet[genbus] == vgens # Pinj = Cg*Pg
    @test cache.qnet[genbus] == vgens # Qinj = Cg*Qg
    @test !iszero(cache)

    return nothing
end

function test_polar_api(polar, device, M)
    pf = polar.network
    tolerance = 1e-8
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    power_balance = ExaPF.PowerFlowBalance(polar) ∘ basis
    # Test that values are matching
    @test myisapprox(pf.vbus, stack.vmag .* exp.(im .* stack.vang))
    xₖ = ExaPF.initial(polar, State())

    # Check that initial residual is correct
    mis = pf.vbus .* conj.(pf.Ybus * pf.vbus) .- pf.sbus
    f_mat = [real(mis[[pf.pv; pf.pq]]); imag(mis[pf.pq])];

    cons = similar(xₖ)
    power_balance(cons, stack)
    @test myisapprox(cons, f_mat)

    # Test powerflow with cache signature
    conv = ExaPF.run_pf(polar, stack)
    @test conv.has_converged

    # Test callbacks
    ## Power Balance
    power_balance(cons, stack)
    # As we run powerflow before, the balance should be below tolerance
    @test ExaPF.xnorm_inf(cons) < tolerance

    ## Cost Production
    cost_production = ExaPF.CostFunction(polar)
    c2 = cost_production(stack)[1]
    @test isa(c2, Real)
    return nothing
end

function test_polar_constraints(polar, device, M)
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    @testset "Expressions $expr" for expr in [
        ExaPF.VoltageMagnitudePQ,
        ExaPF.PowerGenerationBounds,
        ExaPF.LineFlows,
        ExaPF.PowerFlowBalance,
    ]
        # Instantiate
        constraints = expr(polar) ∘ basis
        m = length(constraints)
        @test isa(m, Int)
        g = M{Float64, 1}(undef, m) # TODO: this signature is not great
        fill!(g, 0)
        constraints(g, stack)

        g_min, g_max = ExaPF.bounds(polar, constraints)
        @test length(g_min) == m
        @test length(g_max) == m
        @test isa(g_min, M)
        @test isa(g_max, M)
        @test myisless(g_min, g_max)
    end
    return nothing
end

function test_polar_powerflow(polar, device, M)
    pf_solver = NewtonRaphson(tol=1e-6)
    npartitions = 8
    # Get reduced space Jacobian on the CPU
    J = ExaPF.powerflow_jacobian(polar)
    n = size(J, 1)
    # Build preconditioner
    precond = LS.BlockJacobiPreconditioner(J, npartitions, device)

    J_gpu = ExaPF.powerflow_jacobian_device(polar)

    # Init AD
    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    # Init buffer
    buffer = get(polar, ExaPF.PhysicalState())

    @testset "Powerflow solver $(LinSolver)" for LinSolver in ExaPF.list_solvers(device)
        algo = LinSolver(J_gpu; P=precond)
        ExaPF.init_buffer!(polar, buffer)
        convergence = ExaPF.powerflow(polar, jx, buffer, pf_solver; linear_solver=algo)
        @test convergence.has_converged
        @test convergence.norm_residuals < pf_solver.tol
    end
end
