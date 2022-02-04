module TestPolarFormulation

@eval Base.Experimental.@optlevel 0

using Test

using CUDA
using FiniteDiff
using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays

using ExaPF
import ExaPF: PowerSystem, AutoDiff, LinearSolvers

const PS = PowerSystem
const LS = LinearSolvers

include("api.jl")
include("first_order.jl")
include("second_order.jl")

function myisless(a, b)
    h_a = a |> Array
    h_b = b |> Array
    return h_a <= h_b
end

function myisapprox(a, b; options...)
    h_a = a |> Array
    h_b = b |> Array
    return isapprox(h_a, h_b; options...)
end

function runtests(datafile, device, AT)
    pf = PS.PowerNetwork(datafile)
    polar = PolarForm(pf, device)
    # Test printing
    println(devnull, polar)

    @testset "PolarForm API" begin
        test_polar_api(polar, device, AT)
        test_polar_stack(polar, device, AT)
        test_polar_constraints(polar, device, AT)
        test_polar_powerflow(polar, device, AT)
    end

    @testset "PolarForm AutoDiff (first-order)" begin
        test_constraints_jacobian(polar, device, AT)
        test_constraints_adjoint(polar, device, AT)
        test_full_space_jacobian(polar, device, AT)
        test_reduced_gradient(polar, device, AT)
    end

    @testset "PolarForm AutoDiff (second-order)" begin
        test_hessprod_with_finitediff(polar, device, AT)
        test_full_space_hessian(polar, device, AT)
    end
end

end
