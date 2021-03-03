using CUDA
using CUDA.CUSPARSE
using ExaPF
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs
using FiniteDiff

import ExaPF: AutoDiff
Random.seed!(2713)

const INSTANCES_DIR = joinpath(dirname(@__FILE__), "..", "data")

@testset "ExaPF.PowerSystem" begin
    # Test PowerNetwork object and parsers
    include("powersystem.jl")
end

@testset "ExaPF.LinearSolvers" begin
    include("iterative_solvers.jl")
end

@testset "Polar formulation" begin
    # Test that behavior matches Matpower
    include("Polar/matpower.jl")
    # Test polar formulation
    include("Polar/polar_form.jl")
    # Autodiff's Jacobians
    include("Polar/autodiff.jl")
    # Reduced gradient
    include("Polar/gradient.jl")
    # Reduced Hessian
    include("Polar/hessian.jl")
end

@testset "Optimization evaluators" begin
    # Resolution of powerflow equations with NLPEvaluators
    include("Evaluators/powerflow.jl")
    # Test generic API
    include("Evaluators/interface.jl")
    # Test more in-depth each evaluator
    include("Evaluators/reduced_evaluator.jl")
    include("Evaluators/proxal_evaluator.jl")
    include("Evaluators/auglag.jl")
    include("Evaluators/MOI_wrapper.jl")
    # Test basic reduced gradient algorithm
    include("Evaluators/test_rgm.jl")
end

@testset "Benchmark script" begin
    empty!(ARGS)
    push!(ARGS, "KrylovBICGSTAB")
    push!(ARGS, "CPU")
    push!(ARGS, "case300.m")
    include("../benchmark/benchmarks.jl")
    @test convergence.has_converged
end

