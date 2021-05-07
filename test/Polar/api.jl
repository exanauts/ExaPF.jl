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
    tolerance = 1e-8
    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)
    xₖ = ExaPF.initial(polar, State())
    # Init AD factory
    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())

    # Test powerflow with cache signature
    conv = powerflow(polar, jx, cache, NewtonRaphson(tol=tolerance))
    @test conv.has_converged

    # Get current state
    ExaPF.get!(polar, State(), xₖ, cache)
    # Refresh power of generators in cache
    ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
    ExaPF.update!(polar, PS.Generators(), PS.ReactivePower(), cache)

    # Bounds on state and control
    u_min, u_max = ExaPF.bounds(polar, Control())
    x_min, x_max = ExaPF.bounds(polar, State())

    @test isless(u_min, u_max)
    @test isless(x_min, x_max)

    # Test callbacks
    ## Power Balance
    cons = cache.balance
    ExaPF.power_balance(polar, cons, cache)
    # As we run powerflow before, the balance should be below tolerance
    @test norm(cons, Inf) < tolerance

    ## Cost Production
    c2 = ExaPF.cost_production(polar, cache)
    @test isa(c2, Real)
    return nothing
end

function test_polar_constraints(polar, device, M)
    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)
    ## Inequality constraint
    @testset "Function $cons_function" for cons_function in [
        ExaPF.voltage_magnitude_constraints,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
        ExaPF.flow_constraints,
        ExaPF.power_balance,
    ]
        m = ExaPF.size_constraint(polar, cons_function)
        @test isa(m, Int)
        g = M{Float64, 1}(undef, m) # TODO: this signature is not great
        fill!(g, 0)
        cons_function(polar, g, cache)

        g_min, g_max = ExaPF.bounds(polar, cons_function)
        @test length(g_min) == m
        @test length(g_max) == m
        @test isa(g_min, M)
        @test isa(g_max, M)
        @test g_min <= g_max
    end
    return nothing
end

