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

@testset "PowerSystem parsers" begin
    # Test PSSE parser
    include("powersystem.jl")
    # Test matpower parser
    include("matpower.jl")
end

@testset "Iterative solvers" begin
    include("iterative_solvers.jl")
end

@testset "Resolution of power flow equations" begin
    include("powerflow.jl")
end

@testset "Problem formulations" begin
    include("polar_form.jl")
end

@testset "Optimization evaluators" begin
    include("evaluators.jl")
end

@testset "Reduced space algorithms" begin
    include("reduced_gradient.jl")
    include("test_rgm.jl")
    include("test_ffwu.jl")
end

