using CUDA
using CUDA.CUSPARSE
using CUDAKernels
using ExaPF
using FiniteDiff
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs

import ExaPF: AutoDiff
Random.seed!(2713)

const INSTANCES_DIR = joinpath(dirname(@__FILE__), "..", "data")

# Load test modules
include("test_linear_solvers.jl")
include("Polar/TestPolarForm.jl")

# Define apart tests that are device dependent
function test_suite(device, AT, SMT)
    @testset "Launch tests on $device" begin
        @testset "ExaPF.LinearSolvers on $device" begin
            TestLinearSolvers.runtests(device, AT, SMT)
        end

        @testset "ExaPF.PolarForm on $device" begin
            @testset "case $case" for case in ["case9.m", "case30.m"]
                datafile = joinpath(INSTANCES_DIR, case)
                TestPolarFormulation.runtests(datafile, device, AT)
            end
        end
    end
end

@testset "ExaPF.PowerSystem" begin
    include("powersystem.jl")
end

test_suite(CPU(), Array, SparseMatrixCSC)

if has_cuda_gpu()
    test_suite(CUDADevice(), CuArray, CuSparseMatrixCSR)
end

# @testset "Optimization evaluators" begin
#     # Resolution of powerflow equations with NLPEvaluators
#     include("Evaluators/powerflow.jl")
#     # Test generic API
#     include("Evaluators/interface.jl")
#     # Test more in-depth each evaluator
#     include("Evaluators/reduced_evaluator.jl")
#     include("Evaluators/proxal_evaluator.jl")
#     include("Evaluators/auglag.jl")
#     include("Evaluators/MOI_wrapper.jl")
#     # Test basic reduced gradient algorithm
#     include("Evaluators/test_rgm.jl")
# end

@testset "Documentation" begin
    include("quickstart.jl")
end

@testset "Benchmark script" begin
    empty!(ARGS)
    push!(ARGS, "KrylovBICGSTAB")
    push!(ARGS, "CPU")
    push!(ARGS, "case300.m")
    include("../benchmark/benchmarks.jl")
    @test convergence.has_converged
end

