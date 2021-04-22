module TestPolarFormulation

@eval Base.Experimental.@optlevel 0

using Test

using FiniteDiff
using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays

using ExaPF
import ExaPF: PowerSystem, AutoDiff

const PS = PowerSystem

include("api.jl")
include("autodiff.jl")
include("gradient.jl")
include("hessian.jl")

function runtests(datafile, device, AT)
    pf = PS.PowerNetwork(datafile)
    polar = PolarForm(pf, device)
    # Test printing
    println(devnull, polar)

    @testset "PolarForm API" begin
        test_polar_network_cache(polar, device, AT)
        test_polar_api(polar, device, AT)
        test_polar_constraints(polar, device, AT)
    end

    @testset "PolarForm AutoDiff" begin
        test_constraints_jacobian(polar, device, AT)
        test_constraints_adjoint(polar, device, AT)
    end

    @testset "PolarForm Gradient" begin
        test_reduced_gradient(polar, device, AT)
        test_line_flow_gradient(polar, device, AT)
    end

    @testset "PolarForm Hessians" begin
        test_hessian_with_matpower(polar, device, AT)
        test_hessian_with_finitediff(polar, device, AT)
    end
end

end
