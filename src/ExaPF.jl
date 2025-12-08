module ExaPF

# Standard library
using Printf
using LinearAlgebra
using SparseArrays
import ForwardDiff
import SparseMatrixColorings
using KernelAbstractions
using GPUArraysCore
const KA = KernelAbstractions

import Base: show, get

export run_pf
export State, Control, AllVariables, PolarForm, BlockPolarForm, PolarFormRecourse

# Export KernelAbstractions backends
export CPU

include("templates.jl")
include("utils.jl")

include("autodiff.jl")
using .AutoDiff
include("LinearSolvers/LinearSolvers.jl")
using .LinearSolvers
include("PowerSystem/PowerSystem.jl")
using .PowerSystem

const PS = PowerSystem
const LS = LinearSolvers
const AD = AutoDiff

# Polar formulation
include("Polar/polar.jl")

mutable struct PowerFlowProblem
    form::AbstractFormulation
    stack::AbstractNetworkStack
    powerflow::AD.AbstractExpression
    linear_solver::LS.AbstractLinearSolver
    non_linear_solver::AbstractNonLinearSolver
    mapx::Vector{Int}
    jac::AD.AbstractJacobian
    conv::ConvergenceStatus
    backend::KA.Backend
end

function PowerFlowProblem(
    datafile::String, backend::KA.Backend, formulation::Symbol,
    nscen::Int=1, ploads=nothing, qloads=nothing;
    rtol=1e-8, max_iter=20, verbose=0,
    linear_solver=nothing
)
    form = ExaPF.load_polar(datafile, backend)
    mapx = mapping(form, State())

    stack, jac, powerflow = if formulation == :polar
        nscen == 1 || error("nscen must be 1 for polar formulation")
        @assert nscen == 1 "nscen must be 1 for polar formulation"
        powerflow = ExaPF.PowerFlowBalance(form) ∘ ExaPF.Basis(form);
        stack = ExaPF.NetworkStack(form)
        jac = Jacobian(form, powerflow, mapx)
        @show size(jac.J)
        stack, jac, powerflow
    elseif formulation == :block_polar
        @assert nscen > 1 "nscen must be greater than 1 for block polar formulation"
        blk_form = ExaPF.BlockPolarForm(form, nscen)
        blk_stack = ExaPF.NetworkStack(blk_form)
        blk_powerflow = ExaPF.PowerFlowBalance(blk_form) ∘ ExaPF.Basis(blk_form);
        blk_jac = BatchJacobian(blk_form, blk_powerflow, State())
        @show size(blk_jac.J)
        if isnothing(ploads) || isnothing(qloads)
            @warn "no qloads and ploads for scenarios provided, using random values"
            ploads = rand(get(form, PS.NumberOfBuses()),nscen)
            qloads = rand(get(form, PS.NumberOfBuses()),nscen)
        end
        ExaPF.set_params!(blk_stack, ploads, qloads);
        blk_stack, blk_jac, blk_powerflow
    else
        error("Formulation $formulation not supported")
    end
    if isnothing(linear_solver)
        linear_solver = default_linear_solver(jac.J, backend)
    end
    nlsolver = NewtonRaphson(tol=rtol, maxiter=max_iter, verbose=verbose)
    return PowerFlowProblem(
        form, stack, powerflow,
        linear_solver, nlsolver,
        mapx, jac,
        ConvergenceStatus(false, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0),
        backend
    )
end

get_pd(prob::PowerFlowProblem) = prob.stack.params[1:prob.nbus]
get_qd(prob::PowerFlowProblem) = prob.stack.params[prob.nbus+1:2*prob.nbus]

function set_pd!(prob::PowerFlowProblem, pd::Vector{Float64})
    @assert length(pd) == prob.form.nbus "Length of pd must be equal to the number of buses"
    copyto!(prob.stack.params,0, pd, 0, prob.nbus)
    return prob
end

function set_qd!(prob::PowerFlowProblem, qd::Vector{Float64})
    @assert length(qd) == PS.get(prob.form, PS.NumberOfBuses()) "Length of qd must be equal to the number of buses"
    copyto!(prob.stack.params, prob.nbus, qd, 0, prob.nbus)
    return prob
end

function run_pf(
    datafile::String, backend::KA.Backend=CPU(), formulation::Symbol=:polar, nscen::Int=1,
    ploads=nothing, qloads=nothing; rtol=1e-8, max_iter=20, verbose=0
)
    prob = PowerFlowProblem(
        datafile, backend, formulation, nscen;
        rtol=rtol, max_iter=max_iter, verbose=verbose
    )
    # prob.conv = nlsolve!(prob.non_linear_solver, prob.jac, prob.stack; linear_solver=prob.linear_solver)
    prob.conv = nlsolve!(prob.non_linear_solver, prob.jac, prob.stack)
    return prob
end

solve!(prob::PowerFlowProblem) = nlsolve!(prob.solver, prob.jac, prob.stack; linear_solver=prob.linear_solver)

function show(io::IO, prob::PowerFlowProblem)
    print(io, "PowerFlowProblem\n")
    print(io, "  Formulation: $(prob.form)\n")
    print(io, "  Non-linear solver: $(prob.non_linear_solver)\n")
    print(io, "  Convergence status: $(prob.conv)\n")
    print(io, "  Backend: $(prob.backend)\n")
end

end
