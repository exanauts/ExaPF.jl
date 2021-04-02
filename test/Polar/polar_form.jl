using CUDA
using CUDA.CUSPARSE
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

@testset "Polar formulation" begin
    datafile = joinpath(dirname(@__FILE__), "..", "..", "data", "case9.m")
    tolerance = 1e-8
    pf = PS.PowerNetwork(datafile)

    if has_cuda_gpu()
        ITERATORS = zip([CPU(), CUDADevice()], [Array, CuArray])
    else
        ITERATORS = zip([CPU()], [Array])
    end

    @testset "Initiate polar formulation on device $device" for (device, M) in ITERATORS
        polar = PolarForm(pf, device)
        # Test printing
        println(devnull, polar)

        b = bounds(polar, State())
        b = bounds(polar, Control())

        # Test getters
        nᵤ = ExaPF.get(polar, NumberOfControl())
        nₓ = ExaPF.get(polar, NumberOfState())
        ngen = get(polar, PS.NumberOfGenerators())
        nbus = get(polar, PS.NumberOfBuses())

        # Get initial position
        x0 = ExaPF.initial(polar, State())
        u0 = ExaPF.initial(polar, Control())

        @test length(u0) == nᵤ
        @test length(x0) == nₓ
        for v in [x0, u0]
            @test isa(v, M)
        end

        cache = ExaPF.get(polar, ExaPF.PhysicalState())
        @testset "NetworkState cache" begin
            # By defaut, values are equal to 0
            @test iszero(cache)
            @test isa(cache.vmag, M)
            @test cache.bus_gen == polar.indexing.index_generators
            # Transfer control u0 inside cache
            ExaPF.transfer!(polar, cache, u0)
            # Test that all attributes have valid length
            @test length(cache.vang) == length(cache.vmag) == length(cache.pinj) == length(cache.qinj) == nbus
            @test length(cache.pg) == length(cache.qg) == length(cache.bus_gen) == ngen
            @test length(cache.dx) == length(cache.balance) == nₓ
            # Test setters
            ## Buses
            values = similar(x0, nbus)
            fill!(values, 1)
            ExaPF.setvalues!(cache, PS.VoltageMagnitude(), values)
            @test cache.vmag == values
            ExaPF.setvalues!(cache, PS.VoltageAngle(), values)
            @test cache.vang == values
            ExaPF.setvalues!(cache, PS.ActiveLoad(), values)
            ExaPF.setvalues!(cache, PS.ReactiveLoad(), values)
            # Power generations are still equal to 0, so we get equality
            @test cache.pinj == -values  # Pinj = 0 - Pd
            @test cache.qinj == -values  # Qinj = 0 - Qd
            ## Generators
            vgens = similar(x0, ngen)
            fill!(vgens, 2.0)
            ExaPF.setvalues!(cache, PS.ActivePower(), vgens)
            ExaPF.setvalues!(cache, PS.ReactivePower(), vgens)
            @test cache.pg == vgens
            @test cache.qg == vgens
            genbus = polar.indexing.index_generators
            @test cache.pinj[genbus] == vgens - values[genbus]  # Pinj = Cg*Pg - Pd
            @test cache.qinj[genbus] == vgens - values[genbus]  # Qinj = Cg*Qg - Pd
            # After all these operations, values become non-trivial
            @test !iszero(cache)
        end
        # Reset to default values before going any further
        ExaPF.init_buffer!(polar, cache)
        @test !iszero(cache)

        @testset "Polar model API" begin
            xₖ = copy(x0)
            # Init AD factory
            jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())

            # Test powerflow with cache signature
            conv = powerflow(polar, jx, cache, NewtonRaphson(tol=tolerance))
            # Get current state
            ExaPF.get!(polar, State(), xₖ, cache)
            # Refresh power of generators in cache
            for Power in [PS.ActivePower, PS.ReactivePower]
                ExaPF.update!(polar, PS.Generators(), Power(), cache)
            end

            # Bounds on state and control
            u_min, u_max = ExaPF.bounds(polar, Control())
            x_min, x_max = ExaPF.bounds(polar, State())

            @test isequal(u_min, [0.9, 0.9, 0.9, 0.1, 0.1])
            @test isequal(u_max, [1.1, 1.1, 1.1, 3.0, 2.7])
            @test isequal(x_min, [-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
            @test isequal(x_max, [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])

            # Test callbacks
            ## Power Balance
            cons = cache.balance
            ExaPF.power_balance(polar, cons, cache)
            # As we run powerflow before, the balance should be below tolerance
            @test norm(cons, Inf) < tolerance

            ## Cost Production
            c2 = ExaPF.cost_production(polar, cache)
            @test isa(c2, Real)

            ## Inequality constraint
            for cons_function in [
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
                # Are we on the correct device?
                @test isa(g_min, M)
                @test isa(g_max, M)
                # Test constraints are consistent
                @test g_min <= g_max
            end
        end
    end
end

