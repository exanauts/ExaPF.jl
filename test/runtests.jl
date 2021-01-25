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

import ExaPF: ParsePSSE, PowerSystem, IndexSet

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
    # Reduced gradient
    include("Polar/gradient.jl")
end

@testset "Optimization evaluators" begin
    # Resolution of powerflow equations with NLPEvaluators
    include("Evaluators/powerflow.jl")
    # ReducedSpaceEvaluator API
    include("Evaluators/reduced_evaluator.jl")
    include("Evaluators/proxal_evaluator.jl")
    include("Evaluators/auglag.jl")
    include("Evaluators/MOI_wrapper.jl")
    # Test basic reduced gradient algorithm
    include("Evaluators/test_rgm.jl")
end


