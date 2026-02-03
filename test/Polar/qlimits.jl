# Tests for Q limit enforcement

using ExaPF
using Test
using LinearAlgebra
using LazyArtifacts

const PS = ExaPF.PowerSystem
const INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")

@testset "Q Limit Enforcement" begin
    datafile = joinpath(INSTANCES_DIR, "case9.m")

    @testset "Data structures" begin
        # Test QLimitStatus
        status = ExaPF.QLimitStatus(1, 2, 0.5, :upper)
        @test status.gen_idx == 1
        @test status.bus_idx == 2
        @test status.q_limit == 0.5
        @test status.limit_type == :upper

        # Test display
        io = IOBuffer()
        show(io, status)
        @test occursin("QLimitStatus", String(take!(io)))

        # Test QLimitEnforcementResult
        result = ExaPF.QLimitEnforcementResult(
            true, 2, 15, [status], [0.1, 0.2, 0.3]
        )
        @test result.converged
        @test result.n_outer_iterations == 2
        @test result.n_total_pf_iterations == 15
        @test length(result.violated_generators) == 1
        @test length(result.final_q_values) == 3

        # Test display
        io = IOBuffer()
        show(io, result)
        @test occursin("QLimitEnforcementResult", String(take!(io)))

        # Test BatchedQLimitResult
        batch_result = ExaPF.BatchedQLimitResult([true, false], [result, result])
        @test batch_result.converged[1] == true
        @test batch_result.converged[2] == false

        # Test display
        io = IOBuffer()
        show(io, batch_result)
        @test occursin("BatchedQLimitResult", String(take!(io)))
    end

    @testset "Compute generator reactive power" begin
        polar = ExaPF.PolarForm(datafile, CPU())
        stack = ExaPF.NetworkStack(polar)

        # Run power flow first
        conv = ExaPF.run_pf(polar, stack)
        @test conv.has_converged

        # Compute generator Q
        qgen = ExaPF.compute_generator_reactive_power(polar, stack)
        ngen = get(polar, PS.NumberOfGenerators())
        @test length(qgen) == ngen
        @test all(isfinite.(qgen))

        # Get Q limits and verify computed Q is in reasonable range
        q_min, q_max = PS.bounds(polar.network, PS.Generators(), PS.ReactivePower())
        @test length(q_min) == ngen
        @test length(q_max) == ngen
    end

    @testset "Compute bus reactive power" begin
        polar = ExaPF.PolarForm(datafile, CPU())
        stack = ExaPF.NetworkStack(polar)

        # Run power flow first
        conv = ExaPF.run_pf(polar, stack)
        @test conv.has_converged

        # Compute bus Q
        q_bus = ExaPF.compute_bus_reactive_power(polar, stack)
        nbus = get(polar, PS.NumberOfBuses())
        @test length(q_bus) == nbus
        @test all(isfinite.(q_bus))
    end

    @testset "Check Q violations" begin
        polar = ExaPF.PolarForm(datafile, CPU())
        stack = ExaPF.NetworkStack(polar)

        # Run power flow first
        conv = ExaPF.run_pf(polar, stack)
        @test conv.has_converged

        # Check violations (case9 may or may not have violations)
        bustype = copy(polar.network.bustype)
        violations = ExaPF.check_q_violations(polar, stack, bustype; tol=1e-6)
        @test isa(violations, Vector{ExaPF.QLimitStatus})
    end

    @testset "run_pf with enforce_q_limits" begin
        # Test that run_pf with enforce_q_limits works
        prob = run_pf(datafile; enforce_q_limits=true, verbose=0)
        @test prob.conv.has_converged || !isnothing(prob.qlim_result)

        # Access Q limit result
        result = ExaPF.get_qlimit_result(prob)
        @test !isnothing(result)
        @test isa(result, ExaPF.QLimitEnforcementResult)

        # Test accessor functions
        @test isa(ExaPF.is_qlimit_converged(prob), Bool)
        @test isa(ExaPF.get_violated_generators(prob), Vector{ExaPF.QLimitStatus})
    end

    @testset "Accessor functions" begin
        # Standard power flow (no Q limits)
        prob_std = run_pf(datafile)
        @test prob_std.conv.has_converged
        @test isnothing(ExaPF.get_qlimit_result(prob_std))
        @test ExaPF.is_qlimit_converged(prob_std)  # Returns true for standard PF
        @test isempty(ExaPF.get_violated_generators(prob_std))

        # Q-limited power flow
        prob_qlim = run_pf(datafile; enforce_q_limits=true)

        # Test get_reactive_power_limits
        q_min, q_max = ExaPF.get_reactive_power_limits(prob_qlim)
        ngen = get(prob_qlim.form, PS.NumberOfGenerators())
        @test length(q_min) == ngen
        @test length(q_max) == ngen
        @test all(q_min .<= q_max)

        # Test get_generator_reactive_power
        qgen = ExaPF.get_generator_reactive_power(prob_qlim)
        @test length(qgen) == ngen
        @test all(isfinite.(qgen))

        # Test get_bus_reactive_power
        q_bus = ExaPF.get_bus_reactive_power(prob_qlim)
        nbus = get(prob_qlim.form, PS.NumberOfBuses())
        @test length(q_bus) == nbus

        # Test get_generators_at_limit
        at_qmin, at_qmax = ExaPF.get_generators_at_limit(prob_qlim)
        @test isa(at_qmin, Vector{Int})
        @test isa(at_qmax, Vector{Int})
    end

    @testset "run_pf_with_qlim direct call" begin
        prob = ExaPF.run_pf_with_qlim(datafile, CPU(); verbose=0)
        @test !isnothing(prob.qlim_result)

        result = prob.qlim_result
        @test result.n_outer_iterations >= 1
        @test result.n_total_pf_iterations >= 1
        @test length(result.final_q_values) == get(prob.form, PS.NumberOfGenerators())
    end

    @testset "solve! with enforce_q_limits warning" begin
        prob = PowerFlowProblem(datafile, CPU(), :polar)
        solve!(prob)
        @test prob.conv.has_converged

        # This should emit a warning but still work
        @test_logs (:warn, r"Q limit enforcement for solve!.*") solve!(prob; enforce_q_limits=true)
    end

    @testset "Q limits enforced with guaranteed violations" begin
        # Step 1: Run standard PF to get baseline Q values
        prob_std = run_pf(datafile)
        @test prob_std.conv.has_converged

        qgen_baseline = ExaPF.compute_generator_reactive_power(prob_std.form, prob_std.stack)
        network = prob_std.form.network

        # Step 2: Create modified network with artificially tight Q limits
        data = ExaPF.network_to_data(network)
        GEN_BUS, PG, QG, QMAX, QMIN = PS.IndexSet.idx_gen()[1:5]
        baseMVA = network.baseMVA

        # Find a non-reference bus generator and tighten its limits
        ngen = size(data["gen"], 1)
        modified_gen = 0
        for g in 1:ngen
            bus_idx = network.gen2bus[g]
            if network.bustype[bus_idx] != PS.REF_BUS_TYPE
                q_actual_mva = qgen_baseline[g] * baseMVA
                if q_actual_mva > 0.01
                    # Force upper limit violation: set QMAX below actual
                    data["gen"][g, QMAX] = q_actual_mva * 0.5
                    modified_gen = g
                    break
                elseif q_actual_mva < -0.01
                    # Force lower limit violation: set QMIN above actual
                    data["gen"][g, QMIN] = q_actual_mva * 0.5
                    modified_gen = g
                    break
                end
            end
        end
        @test modified_gen > 0  # Ensure we modified something

        # Step 3: Create modified network and run Q-limited PF
        modified_network = PS.PowerNetwork(data)
        prob_qlim = ExaPF.run_pf_with_qlim(modified_network, CPU(); verbose=0)

        # Step 4: Verify violations were detected
        result = prob_qlim.qlim_result
        @test !isnothing(result)
        @test !isempty(result.violated_generators)
        @test result.n_outer_iterations > 1  # Had to iterate due to violations

        # Step 5: Verify final Q values are within limits
        qgen_final = ExaPF.get_generator_reactive_power(prob_qlim)
        q_min, q_max = ExaPF.get_reactive_power_limits(prob_qlim)
        tol = 1e-5
        @test all(qgen_final .>= q_min .- tol)
        @test all(qgen_final .<= q_max .+ tol)
    end
end
