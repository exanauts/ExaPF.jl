using CUDA
using CUDA.CUSPARSE
using ExaPF
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs

import ExaPF: ParsePSSE, PowerSystem, IndexSet

Random.seed!(2713)

@testset "Problem formulations" begin
    # Test PowerNetwork object and parsers
    include("powersystem.jl")
    # Test that behavior matches Matpower
    include("matpower.jl")
    # Test polar formulation
    include("polar_form.jl")
end

@testset "Iterative solvers" begin
    include("iterative_solvers.jl")
end

@testset "Optimization evaluators" begin
    # Resolution of powerflow equations with NLPEvaluators
    include("powerflow.jl")
    # Reduced gradient
    include("reduced_gradient.jl")
    # ReducedSpaceEvaluator API
    include("evaluators.jl")
end

@testset "Reduced space algorithms" begin
    include("test_rgm.jl")
    # include("test_ffwu.jl")
end

