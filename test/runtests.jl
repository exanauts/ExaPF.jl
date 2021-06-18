using Test
using Random
using LinearAlgebra
using SparseArrays

using CUDA
using KernelAbstractions

using FiniteDiff

using ExaPF

Random.seed!(2713)

const INSTANCES_DIR = joinpath(dirname(@__FILE__), "..", "data")
const BENCHMARK_DIR = joinpath(dirname(@__FILE__), "..", "benchmark")
const CASES = ["case9.m", "case30.m"]

ARCHS = Any[(CPU(), Array, SparseMatrixCSC)]
if has_cuda_gpu()
    push!(ARCHS, (CUDADevice(), CuArray, CuSparseMatrixCSR))
    include("cusolver.jl")
end

# Load test modules
@isdefined(TestLinearSolvers)    || include("TestLinearSolvers.jl")
@isdefined(TestPolarFormulation) || include("Polar/TestPolarForm.jl")
@isdefined(TestEvaluators)       || include("Evaluators/TestEvaluators.jl")

init_time = time()
@testset "Test ExaPF" begin
    @testset "ExaPF.PowerSystem" begin
        @info "Test PowerSystem submodule ..."
        tic = time()
        include("powersystem.jl")
        println("Took $(round(time() - tic; digits=1)) seconds.")

        @info "Compare power flow with MATPOWER ..."
        tic = time()
        include("Polar/matpower.jl")
        println("Took $(round(time() - tic; digits=1)) seconds.")
    end
    println()

    @testset "Test device specific code on $device" for (device, AT, SMT) in ARCHS
        @info "Test device $device"

        println("Test LinearSolvers submodule ...")
        tic = time()
        @testset "ExaPF.LinearSolvers" begin
            TestLinearSolvers.runtests(device, AT, SMT)
        end
        println("Took $(round(time() - tic; digits=1)) seconds.")

        println("Test PolarForm ...")
        tic = time()
        @testset "ExaPF.PolarForm ($case)" for case in CASES
            datafile = joinpath(INSTANCES_DIR, case)
            TestPolarFormulation.runtests(datafile, device, AT)
        end
        println("Took $(round(time() - tic; digits=1)) seconds.")

        println("Test Evaluators ...")
        tic = time()
        @testset "ExaPF.Evaluator $(case)" for case in CASES
            datafile = joinpath(INSTANCES_DIR, case)
            TestEvaluators.runtests(datafile, device, AT)
        end
        println("Took $(round(time() - tic; digits=1)) seconds.")
    end
    println()

    @testset "Test reduced gradient algorithms" begin
        @info "Test reduced gradient algorithm ..."
        tic = time()
        include("Evaluators/test_rgm.jl")
        include("Evaluators/MOI_wrapper.jl")
        println("Took $(round(time() - tic; digits=1)) seconds.\n")
    end

    @testset "Test Documentation" begin
        include("quickstart.jl")
    end

    @testset "Test Benchmark script" begin
        empty!(ARGS)
        push!(ARGS, "KrylovBICGSTAB")
        push!(ARGS, "CPU")
        push!(ARGS, "case300.m")
        include(joinpath(BENCHMARK_DIR, "benchmarks.jl"))
    end
end
println("TOTAL RUNNING TIME: $(round(time() - init_time; digits=1)) seconds.")

