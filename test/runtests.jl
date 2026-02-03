using LinearAlgebra
using Random
using SparseArrays
using Test

using KernelAbstractions

using ExaPF

Random.seed!(2713)

const BENCHMARK_DIR = joinpath(dirname(@__FILE__), "..", "benchmark")
const EXAMPLES_DIR = joinpath(dirname(@__FILE__), "..", "examples")
const CASES = ["case9.m", "case30.m"]

# Load GPU backends dynamically
include("setup.jl")

# Load test modules
@isdefined(TestKernels)          || include("TestKernels.jl")
@isdefined(TestLinearSolvers)    || include("TestLinearSolvers.jl")
@isdefined(TestPolarFormulation) || include("Polar/TestPolarForm.jl")
@isdefined(ExaBenchmark)         || include(joinpath(BENCHMARK_DIR, "benchmarks.jl"))

init_time = time()
@testset "Test ExaPF" begin
    @testset "ExaPF.PowerSystem" begin
        @info "Test PowerSystem submodule ..."
        tic = time()
        include("powersystem.jl")
        println("Took $(round(time() - tic; digits=1)) seconds.")

        @info "Test kernels ..."
        tic = time()
        TestKernels.runtests(CPU(), Array, SparseMatrixCSC)
        println("Took $(round(time() - tic; digits=1)) seconds.")

        @info "Compare power flow with MATPOWER ..."
        tic = time()
        include("Polar/matpower.jl")
        println("Took $(round(time() - tic; digits=1)) seconds.")
    end
    println()

    @testset "Test backend specific code on $backend" for (backend, AT, SMT, arch) in ARCHS
        @info "Test backend $backend"

        println("Test LinearSolvers submodule ...")
        tic = time()
        @testset "ExaPF.LinearSolvers" begin
            TestLinearSolvers.runtests(backend, AT, SMT)
        end
        println("Took $(round(time() - tic; digits=1)) seconds.")

        println("Test PolarForm ...")
        tic = time()
        @testset "ExaPF.PolarForm ($case)" for case in CASES
            TestPolarFormulation.runtests(case, backend, AT, arch)
        end
        println("Took $(round(time() - tic; digits=1)) seconds.")
    end
    println()

    @testset "Batched PowerFlow" begin
        @info "Test batched powerflow across all backends ..."
        tic = time()
        include("batch_powerflow.jl")
        println("Took $(round(time() - tic; digits=1)) seconds.")
    end
    println()

    @testset "Q Limit Enforcement" begin
        @info "Test Q limit enforcement ..."
        tic = time()
        include("Polar/qlimits.jl")
        println("Took $(round(time() - tic; digits=1)) seconds.")
    end
    println()

    include("quickstart.jl")

    @testset "Benchmark" begin
        @info("Test benchmark suite")
        ExaBenchmark.benchmark("case30.m", CPU())
    end
    @testset "Test example" begin
        @info("Test example")
        for file in filter(
            f -> endswith(f, ".jl"),
            readdir(EXAMPLES_DIR),
        )
            include(joinpath(EXAMPLES_DIR, file))
        end
    end
    println()
end
println("TOTAL RUNNING TIME: $(round(time() - init_time; digits=1)) seconds.")

