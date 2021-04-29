module TestEvaluators

@eval Base.Experimental.@optlevel 0

using Test

using FiniteDiff
using LinearAlgebra
using Random
using SparseArrays
using KernelAbstractions

using ExaPF
import ExaPF: PowerSystem, LinearSolvers

const PS = PowerSystem
const LS = LinearSolvers

include("powerflow.jl")
include("api.jl")
include("proxal_evaluator.jl")
include("auglag.jl")

function _init(datafile, ::Type{ExaPF.ReducedSpaceEvaluator}, device)
    constraints = Function[
        ExaPF.voltage_magnitude_constraints,
        ExaPF.reactive_power_constraints,
        ExaPF.active_power_constraints,
        # ExaPF.flow_constraints,
    ]
    return ExaPF.ReducedSpaceEvaluator(datafile; device=device, constraints=constraints)
end
function _init(datafile, ::Type{ExaPF.ProxALEvaluator}, device)
    nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=device)
    time = ExaPF.Normal
    return ExaPF.ProxALEvaluator(nlp, time)
end
function _init(datafile, ::Type{ExaPF.SlackEvaluator}, device)
    return ExaPF.SlackEvaluator(datafile; device=device)
end

function runtests(datafile, device, AT)
    @testset "Newton-Raphson resolution" begin
        nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=device, powerflow_solver=NewtonRaphson(tol=1e-6))
        test_powerflow_evaluator(nlp, device, AT)
    end
    @testset "$Evaluator Interface" for Evaluator in [
        ExaPF.ReducedSpaceEvaluator,
        ExaPF.AugLagEvaluator,
        ExaPF.ProxALEvaluator,
        ExaPF.SlackEvaluator,
        ExaPF.FeasibilityEvaluator,
    ]
        nlp = Evaluator(datafile; device=device)
        test_evaluator_api(nlp, device, AT)
        test_evaluator_callbacks(nlp, device, AT)
    end
    @testset "$Evaluator Hessian" for Evaluator in [
        ExaPF.ReducedSpaceEvaluator,
        ExaPF.AugLagEvaluator,
    ]
        nlp = Evaluator(datafile; device=device)
        test_evaluator_hessian(nlp, device, AT)
    end
    @testset "ProxALEvaluator" begin
        nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=device)
        test_proxal_evaluator(nlp, device, AT)
    end
    @testset "AugLagEvaluator with $Evaluator backend" for Evaluator in [
        ExaPF.ReducedSpaceEvaluator,
        ExaPF.ProxALEvaluator,
        ExaPF.SlackEvaluator,
    ]
        nlp = _init(datafile, Evaluator, device)
        test_auglag_evaluator(nlp, device, AT)
    end
end

end
