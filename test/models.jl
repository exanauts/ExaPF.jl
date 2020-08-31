using Revise
using Printf
using BenchmarkTools
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs
using ExaPF
import ExaPF: PowerSystem, AD, Precondition, Iterative
include("../src/models/models.jl")

# TODO: for some reason, convergence is non-deterministic if we do not
# fix a seed
Random.seed!(27)

@testset "Formulation" begin
    datafile = "test/data/case9.m"
    pf = PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, CPU())
    println(typeof(polar))

    b = bounds(polar, State())
    b = bounds(polar, Control())

    nᵤ = get(polar, NumberOfControl())
    nₓ = get(polar, NumberOfState())

    x0 = initial(polar, State())
    u0 = initial(polar, Control())
    p = initial(polar, Parameters())

    @test length(u0) == nᵤ
    @test length(x0) == nₓ

    @time powerflow(polar, x0, u0, p)
    @time powerflow(polar, x0, u0, p)
end
