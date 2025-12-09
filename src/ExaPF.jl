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
export PowerFlowProblem
export run_pf, get_sol, set_pd!, set_qd!, get_pd, get_qd

"""
    PowerFlowProblem

A mutable struct representing a power flow problem.

# Fields
- `form::AbstractFormulation`: The problem formulation (e.g., PolarForm, BlockPolarForm)
- `stack::AbstractNetworkStack`: Stack containing network variables and parameters
- `powerflow::AD.AbstractExpression`: Power flow balance expression
- `linear_solver::LS.AbstractLinearSolver`: Linear solver for Newton-Raphson iterations
- `non_linear_solver::AbstractNonLinearSolver`: Non-linear solver configuration
- `mapx::Vector{Int}`: Mapping indices for state variables
- `jac::AD.AbstractJacobian`: Jacobian matrix for automatic differentiation
- `conv::ConvergenceStatus`: Convergence status of the solver
- `backend::KA.Backend`: Computation backend (CPU or GPU)

# See also
[`run_pf`](@ref), [`solve!`](@ref)
"""
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

"""
    PowerFlowProblem(datafile, backend, formulation, nscen=1, ploads=nothing, qloads=nothing;
                     rtol=1e-8, max_iter=20, verbose=0, linear_solver=nothing)

Construct a `PowerFlowProblem` from a data file.

# Arguments
- `datafile::String`: Path to the power system data file (e.g., MATPOWER format)
- `backend::KA.Backend`: Computation backend (CPU() or GPU backend)
- `formulation::Symbol`: Problem formulation type (`:polar` or `:block_polar`)
- `nscen::Int=1`: Number of scenarios (must be 1 for `:polar`, >1 for `:block_polar`)
- `ploads=nothing`: Active power loads for scenarios (required for `:block_polar` with nscen>1)
- `qloads=nothing`: Reactive power loads for scenarios (required for `:block_polar` with nscen>1)

# Keyword Arguments
- `rtol=1e-8`: Relative tolerance for convergence
- `max_iter=20`: Maximum number of Newton-Raphson iterations
- `verbose=0`: Verbosity level (0=silent, higher values=more output)
- `linear_solver=nothing`: Custom linear solver (if `nothing`, uses default for backend)

# Returns
- `PowerFlowProblem`: A configured power flow problem ready to be solved

# Examples
```julia
# Single scenario polar formulation
prob = PowerFlowProblem("case9.m", CPU(), :polar)

# Multiple scenario block polar formulation
nscen = 100
ploads = rand(9, nscen)  # Random loads for 9 buses
qloads = rand(9, nscen)
prob = PowerFlowProblem("case9.m", CPU(), :block_polar, nscen, ploads, qloads)
```

# See also
[`run_pf`](@ref), [`solve!`](@ref)
"""
function PowerFlowProblem(
    datafile::String, backend::KA.Backend, formulation::Symbol,
    nscen::Int=1, ploads=nothing, qloads=nothing;
    rtol=1e-8, max_iter=20, verbose=0,
    linear_solver=nothing, batch_linear_solver=false
)
    form = ExaPF.load_polar(datafile, backend)
    mapx = mapping(form, State())

    stack, jac, powerflow = if formulation == :polar
        nscen == 1 || error("nscen must be 1 for polar formulation")
        @assert nscen == 1 "nscen must be 1 for polar formulation"
        powerflow = ExaPF.PowerFlowBalance(form) ∘ ExaPF.Basis(form);
        stack = ExaPF.NetworkStack(form)
        jac = Jacobian(form, powerflow, mapx)
        ExaPF.set_params!(blk_jac, blk_stack);
        ExaPF.jacobian!(blk_jac, blk_stack);
        stack, jac, powerflow
    elseif formulation == :block_polar
        @assert nscen > 1 "nscen must be greater than 1 for block polar formulation"
        blk_form = ExaPF.BlockPolarForm(form, nscen)
        blk_stack = ExaPF.NetworkStack(blk_form)
        blk_powerflow = ExaPF.PowerFlowBalance(blk_form) ∘ ExaPF.Basis(blk_form);
        blk_jac = BatchJacobian(blk_form, blk_powerflow, State())
        if isnothing(ploads) || isnothing(qloads)
            @warn "no qloads and ploads for scenarios provided, using random values"
            ploads = rand(get(form, PS.NumberOfBuses()),nscen)
            qloads = rand(get(form, PS.NumberOfBuses()),nscen)
        end
        ExaPF.set_params!(blk_stack, ploads, qloads);
        ExaPF.set_params!(blk_jac, blk_stack);
        ExaPF.jacobian!(blk_jac, blk_stack);
        blk_stack, blk_jac, blk_powerflow
    else
        error("Formulation $formulation not supported")
    end
    if isnothing(linear_solver)
        linear_solver = default_linear_solver(jac, backend)
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

"""
    get_pd(prob::PowerFlowProblem)

Get the active power demand (Pd) values from the power flow problem.

# Arguments
- `prob::PowerFlowProblem`: The power flow problem instance

# Returns
- `Vector`: Active power demand values for all buses in the system

# See also
[`get_qd`](@ref), [`set_pd!`](@ref)
"""
get_pd(prob::PowerFlowProblem) = prob.stack.params[1:prob.nbus]

"""
    get_qd(prob::PowerFlowProblem)

Get the reactive power demand (Qd) values from the power flow problem.

# Arguments
- `prob::PowerFlowProblem`: The power flow problem instance

# Returns
- `Vector`: Reactive power demand values for all buses in the system

# See also
[`get_pd`](@ref), [`set_qd!`](@ref)
"""
get_qd(prob::PowerFlowProblem) = prob.stack.params[prob.nbus+1:2*prob.nbus]

"""
    set_pd!(prob::PowerFlowProblem, pd::Vector{Float64})

Set the active power demand (Pd) values for the power flow problem.

This function modifies the problem in-place, updating the active power demand
parameters in the network stack.

# Arguments
- `prob::PowerFlowProblem`: The power flow problem instance to modify
- `pd::Vector{Float64}`: New active power demand values for all buses

# Returns
- `PowerFlowProblem`: The modified problem instance

# Throws
- `AssertionError`: If the length of `pd` does not match the number of buses

# Examples
```julia
prob = PowerFlowProblem("case9.m", CPU(), :polar)
pd_new = ones(9) * 0.5  # Set all buses to 0.5 p.u.
set_pd!(prob, pd_new)
```

# See also
[`set_qd!`](@ref), [`get_pd`](@ref)
"""
function set_pd!(prob::PowerFlowProblem, pd::Vector{Float64})
    @assert length(pd) == prob.form.nbus "Length of pd must be equal to the number of buses"
    copyto!(prob.stack.params,0, pd, 0, prob.nbus)
    return prob
end

"""
    set_qd!(prob::PowerFlowProblem, qd::Vector{Float64})

Set the reactive power demand (Qd) values for the power flow problem.

This function modifies the problem in-place, updating the reactive power demand
parameters in the network stack.

# Arguments
- `prob::PowerFlowProblem`: The power flow problem instance to modify
- `qd::Vector{Float64}`: New reactive power demand values for all buses

# Returns
- `PowerFlowProblem`: The modified problem instance

# Throws
- `AssertionError`: If the length of `qd` does not match the number of buses

# Examples
```julia
prob = PowerFlowProblem("case9.m", CPU(), :polar)
qd_new = ones(9) * 0.2  # Set all buses to 0.2 p.u.
set_qd!(prob, qd_new)
```

# See also
[`set_pd!`](@ref), [`get_qd`](@ref)
"""

get_sol(prob::PowerFlowProblem) = prob.stack.input[prob.jac.map]
function set_qd!(prob::PowerFlowProblem, qd::Vector{Float64})
    @assert length(qd) == PS.get(prob.form, PS.NumberOfBuses()) "Length of qd must be equal to the number of buses"
    copyto!(prob.stack.params, prob.nbus, qd, 0, prob.nbus)
    return prob
end

"""
    run_pf(datafile, backend=CPU(), formulation=:polar, nscen=1, ploads=nothing, qloads=nothing;
           rtol=1e-8, max_iter=20, verbose=0)

Solve a power flow problem from a data file.

This is a convenience function that constructs a `PowerFlowProblem` and immediately
solves it using the Newton-Raphson method.

# Arguments
- `datafile::String`: Path to the power system data file
- `backend::KA.Backend=CPU()`: Computation backend (CPU() or GPU backend)
- `formulation::Symbol=:polar`: Problem formulation (`:polar` or `:block_polar`)
- `nscen::Int=1`: Number of scenarios
- `ploads=nothing`: Active power loads for scenarios (for `:block_polar`)
- `qloads=nothing`: Reactive power loads for scenarios (for `:block_polar`)

# Keyword Arguments
- `rtol=1e-8`: Relative tolerance for convergence
- `max_iter=20`: Maximum number of iterations
- `verbose=0`: Verbosity level

# Returns
- `PowerFlowProblem`: The solved power flow problem with convergence information

# Examples
```julia
# Solve a single scenario power flow
prob = run_pf("case9.m")

# Solve with custom tolerance and verbosity
prob = run_pf("case9.m", CPU(), :polar; rtol=1e-10, verbose=1)

# Solve multiple scenarios
nscen = 50
ploads = rand(9, nscen)
qloads = rand(9, nscen)
prob = run_pf("case9.m", CPU(), :block_polar, nscen, ploads, qloads)
```

# See also
[`PowerFlowProblem`](@ref), [`solve!`](@ref)
"""
function run_pf(
    datafile::String, backend::KA.Backend=CPU(),
    formulation::Symbol=:polar, nscen::Int=1,
    ploads=nothing, qloads=nothing; rtol=1e-8,
    max_iter=20, verbose=0, batch_linear_solver=false,
)
    prob = PowerFlowProblem(
        datafile, backend, formulation, nscen,
        ploads, qloads;
        rtol=rtol, max_iter=max_iter, verbose=verbose,
        batch_linear_solver=batch_linear_solver
    )
    # prob.conv = nlsolve!(prob.non_linear_solver, prob.jac, prob.stack; linear_solver=prob.linear_solver)
    prob.conv = nlsolve!(prob.non_linear_solver, prob.jac, prob.stack)
    return prob
end

"""
    solve!(prob::PowerFlowProblem)

Re-solve an existing power flow problem.

This function re-runs the non-linear solver on the problem, which is useful after
modifying problem parameters (e.g., via `set_pd!` or `set_qd!`).

# Arguments
- `prob::PowerFlowProblem`: The power flow problem to solve

# Returns
- Convergence status of the solver

# Examples
```julia
# Create and solve initial problem
prob = PowerFlowProblem("case9.m", CPU(), :polar)
solve!(prob)

# Modify parameters and re-solve
set_pd!(prob, ones(9) * 0.8)
solve!(prob)
```

# See also
[`run_pf`](@ref), [`set_pd!`](@ref), [`set_qd!`](@ref)
"""
solve!(prob::PowerFlowProblem) = nlsolve!(prob.solver, prob.jac, prob.stack; linear_solver=prob.linear_solver)

"""
    show(io::IO, prob::PowerFlowProblem)

Display a summary of the power flow problem.

Prints information about the formulation, solver configuration, convergence status,
and computational backend.

# Arguments
- `io::IO`: Output stream
- `prob::PowerFlowProblem`: The power flow problem to display
"""
function show(io::IO, prob::PowerFlowProblem)
    print(io, "PowerFlowProblem\n")
    print(io, "  Formulation: $(prob.form)\n")
    print(io, "  Non-linear solver: $(prob.non_linear_solver)\n")
    print(io, "  Convergence status: $(prob.conv)\n")
    print(io, "  Backend: $(prob.backend)\n")
end

end
