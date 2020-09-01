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

    jx, ju = init_ad_factory(polar, x0, u0, p)
    @time powerflow(polar, jx, x0, u0, p)
    @time powerflow(polar, jx, x0, u0, p, verbose_level=0)
end
@testset "Formulation" begin
    datafile = "test/data/case9.m"
    pf = PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, CUDADevice())
    nᵤ = get(polar, NumberOfControl())
    nₓ = get(polar, NumberOfState())

    x0 = initial(polar, State())
    u0 = initial(polar, Control())
    p = initial(polar, Parameters())

    @test length(u0) == nᵤ
    @test length(x0) == nₓ
    @test isa(u0, CuArray)
    @test isa(x0, CuArray)
    jx, ju = init_ad_factory(polar, x0, u0, p)
    @time powerflow(polar, jx, x0, u0, p, verbose_level=0)
    @time powerflow(polar, jx ,x0, u0, p, verbose_level=0)
end
