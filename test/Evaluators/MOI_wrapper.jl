
using Ipopt
using MathOptInterface

const MOI = MathOptInterface

@testset "MOI wrapper" begin
    CASE57_SOLUTION = [
        1.0260825400262428,
        1.016218932360429,
        1.009467718490475,
        1.027857273911734,
        1.06,
        0.9962215167521776,
        0.9965415804936182,
        0.0,
        0.6,
        0.0,
        8.603379123083476,
        0.0,
        1.3982297290334753,
    ]

    datafile = joinpath(INSTANCES_DIR, "case57.m")
    nlp = ExaPF.ReducedSpaceEvaluator(datafile)

    @testset "ReducedSpaceEvaluator with L-BFGS" begin
        optimizer = Ipopt.Optimizer()
        MOI.set(optimizer, MOI.RawParameter("print_level"), 0)
        MOI.set(optimizer, MOI.RawParameter("limited_memory_max_history"), 50)
        MOI.set(optimizer, MOI.RawParameter("hessian_approximation"), "limited-memory")
        MOI.set(optimizer, MOI.RawParameter("tol"), 1e-4)

        solution = ExaPF.optimize!(optimizer, nlp)
        MOI.empty!(optimizer)
        @test solution.status ∈ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        @test solution.minimum ≈ 3.7589338e+04
        @test solution.minimizer ≈ CASE57_SOLUTION rtol=1e-5
    end

    # Test resolution with AugLagEvaluator and Ipopt, as used inside ProxAL
    @testset "AugLagEvaluator with Hessian" begin
        optimizer = Ipopt.Optimizer()
        MOI.set(optimizer, MOI.RawParameter("print_level"), 0)
        MOI.set(optimizer, MOI.RawParameter("tol"), 1e-4)
        MOI.set(optimizer, MOI.RawParameter("max_iter"), 30)
        aug = ExaPF.AugLagEvaluator(nlp, ExaPF.initial(nlp))

        solution = ExaPF.optimize!(optimizer, aug)
        MOI.empty!(optimizer)
        @test solution.status ∈ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        @test solution.minimum ≈ 25138.76310420395
    end
end

