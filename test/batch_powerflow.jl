using LinearAlgebra
using Test

# Test batched powerflow with all available backends
@testset "Batched PowerFlow" begin
    case = "case9.m"
    nscen = 10

    # Load polar form and create load matrices
    polar = ExaPF.load_polar(case)
    nbus = ExaPF.PowerSystem.get(polar, ExaPF.PowerSystem.NumberOfBuses())
    stack = ExaPF.NetworkStack(polar)
    pload = stack.params[1:nbus]
    qload = stack.params[nbus+1:2*nbus]
    ploads = repeat(pload, 1, nscen)
    qloads = repeat(qload, 1, nscen)

    # Compute single scenario solution once on CPU
    @info "Computing reference single scenario solution on CPU"
    res_single = run_pf(case, CPU(), :polar; verbose=0)
    sol_single = get_solution(res_single)
    norm_single = norm(sol_single)
    nstates = length(sol_single)  # Number of state variables per scenario

    # Test batched powerflow on all backends
    @testset "Backend: $arch" for (backend, AT, SMT, arch) in ARCHS
        @info "Testing batched powerflow on $arch"

        # Run batched powerflow
        res_batch = run_pf(case, backend, :block_polar, nscen, ploads, qloads; verbose=0)
        sol_batch = get_solution(res_batch)

        # Check solution dimensions
        @test length(sol_batch) == nstates * nscen

        # Verify each scenario matches the single scenario solution norm
        for i in 1:nscen
            scenario_sol = sol_batch[(i-1)*nstates + 1 : i*nstates]
            norm_scenario = norm(scenario_sol)
            @test isapprox(norm_scenario, norm_single, rtol=1e-8)
        end

        @info "✓ All $nscen scenarios match reference solution on $arch"
    end
end
