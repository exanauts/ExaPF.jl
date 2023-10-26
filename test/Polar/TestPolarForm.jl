module TestPolarFormulation

using Test

using AMDGPU
using CUDA
using FiniteDiff
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
include("recourse.jl")

function myisless(a, b)
    h_a = a |> Array
    h_b = b |> Array
    return h_a <= h_b
end

function myisapprox(a, b; options...)
    h_a = a |> Array
    h_b = b |> Array
    istrue = isapprox(h_a, h_b; options...)
    if istrue
        return istrue
    else
        println(h_a)
        println(h_b)
        return istrue
    end
end

function runtests(case, device, AT)
    polar = ExaPF.load_polar(case, device)
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

    @testset "BlockPolarForm" begin
        test_block_stack(polar, device, AT)
        test_block_expressions(polar, device, AT)
        test_block_powerflow(polar, device, AT)
        test_block_jacobian(polar, device, AT)
        test_block_hessian(polar, device, AT)
    end

    @testset "Contingency" begin
        test_contingency_powerflow(polar, device, AT)
    end

    @testset "PolarFormRecourse" begin
        test_recourse_expression(polar, device, AT)
        test_recourse_powerflow(polar, device, AT)
        if isa(device, CPU)
            test_recourse_jacobian(polar, device, AT)
            test_recourse_hessian(polar, device, AT)
            test_recourse_block_hessian(polar, device, AT)
        end
    end
end

end
