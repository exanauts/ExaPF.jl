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
export run_pf, get_solution, set_active_load!, set_reactive_load!, get_active_load, get_reactive_load, get_convergence_status
export get_voltage_magnitude, get_voltage_angle, solve!

# Q limit enforcement
export QLimitStatus, QLimitEnforcementResult, BatchedQLimitResult
export compute_generator_reactive_power, compute_bus_reactive_power, check_q_violations
export get_reactive_power_limits, get_generator_reactive_power, get_qlimit_result
export get_violated_generators, is_qlimit_converged, get_bus_reactive_power, get_generators_at_limit
export run_pf_with_qlim

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
- `qlim_result::Union{QLimitEnforcementResult, Nothing}`: Q limit enforcement result (if enabled)

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
    qlim_result::Union{QLimitEnforcementResult, Nothing}
end

"""
    PowerFlowProblem(datafile, backend, formulation, nscen=1, ploads=nothing, qloads=nothing;
                     rtol=1e-8, max_iter=20, verbose=0, linear_solver=nothing, batch_linear_solver=false)

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
    linear_solver=nothing,
)
    form = ExaPF.load_polar(datafile, backend)
    mapx = mapping(form, State())

    stack, jac, powerflow = if formulation == :polar
        nscen == 1 || error("nscen must be 1 for polar formulation")
        @assert nscen == 1 "nscen must be 1 for polar formulation"
        powerflow = ExaPF.PowerFlowBalance(form) ∘ ExaPF.Basis(form);
        stack = ExaPF.NetworkStack(form)
        jac = Jacobian(form, powerflow, mapx)
        ExaPF.set_params!(jac, stack);
        ExaPF.jacobian!(jac, stack);
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
        linear_solver = default_linear_solver(jac.J; nblocks=nscen)
    end
    nlsolver = NewtonRaphson(tol=rtol, maxiter=max_iter, verbose=verbose)
    return PowerFlowProblem(
        form, stack, powerflow,
        linear_solver, nlsolver,
        mapx, jac,
        ConvergenceStatus(false, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0),
        backend,
        nothing  # qlim_result
    )
end

"""
    get_active_load(prob::PowerFlowProblem)

Get the active power demand (Pd) values from the power flow problem.

# Arguments
- `prob::PowerFlowProblem`: The power flow problem instance

# Returns
- `Vector`: Active power demand values for all buses in the system

# See also
[`get_reactive_load`](@ref), [`set_active_load!`](@ref)
"""
get_active_load(prob::PowerFlowProblem) = prob.stack.params[1:prob.nbus]

"""
    get_reactive_load(prob::PowerFlowProblem)

Get the reactive power demand (Qd) values from the power flow problem.

# Arguments
- `prob::PowerFlowProblem`: The power flow problem instance

# Returns
- `Vector`: Reactive power demand values for all buses in the system

# See also
[`get_active_load`](@ref), [`set_reactive_load!`](@ref)
"""
get_reactive_load(prob::PowerFlowProblem) = prob.stack.params[prob.nbus+1:2*prob.nbus]

"""
    set_active_load!(prob::PowerFlowProblem, pd::Vector{Float64})

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
set_active_load!(prob, pd_new)
```

# See also
[`set_reactive_load!`](@ref), [`get_active_load`](@ref)
"""
function set_active_load!(prob::PowerFlowProblem, pd::Vector{Float64})
    @assert length(pd) == prob.form.nbus "Length of pd must be equal to the number of buses"
    copyto!(prob.stack.params,0, pd, 0, prob.nbus)
    return prob
end

"""
    set_reactive_load!(prob::PowerFlowProblem, qd::Vector{Float64})

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
set_reactive_load!(prob, qd_new)
```

# See also
[`set_active_load!`](@ref), [`get_reactive_load`](@ref)
"""
function set_reactive_load!(prob::PowerFlowProblem, qd::Vector{Float64})
    @assert length(qd) == PS.get(prob.form, PS.NumberOfBuses()) "Length of qd must be equal to the number of buses"
    copyto!(prob.stack.params, prob.nbus, qd, 0, prob.nbus)
    return prob
end

function get_convergence_status(prob::PowerFlowProblem)
    return prob.conv
end

"""
    get_voltage_angle(prob::PowerFlowProblem)

Get the voltage angle values from the power flow problem solution.

# Arguments
- `prob::PowerFlowProblem`: The power flow problem instance

# Returns
- `Vector`: Voltage angle values (in radians) for all buses in the system

# See also
[`get_voltage_magnitude`](@ref), [`get_solution`](@ref)
"""
get_voltage_angle(prob::PowerFlowProblem) = prob.stack.vang

"""
    get_voltage_magnitude(prob::PowerFlowProblem)

Get the voltage magnitude values from the power flow problem solution.

# Arguments
- `prob::PowerFlowProblem`: The power flow problem instance

# Returns
- `Vector`: Voltage magnitude values (in per-unit) for all buses in the system

# See also
[`get_voltage_angle`](@ref), [`get_solution`](@ref)
"""
get_voltage_magnitude(prob::PowerFlowProblem) = prob.stack.vmag

"""
    get_solution(prob::PowerFlowProblem)

Get the complete solution vector from the power flow problem.

This function returns the state variables (voltage angles and magnitudes) that
were solved by the Newton-Raphson method, in the order specified by the mapping
indices.

# Arguments
- `prob::PowerFlowProblem`: The power flow problem instance

# Returns
- `Vector`: Solution vector containing the state variables

# See also
[`get_voltage_angle`](@ref), [`get_voltage_magnitude`](@ref), [`solve!`](@ref)
"""
get_solution(prob::PowerFlowProblem) = prob.stack.input[prob.jac.map]

"""
    run_pf(datafile, backend=CPU(), formulation=:polar, nscen=1, ploads=nothing, qloads=nothing;
           rtol=1e-8, max_iter=20, verbose=0, enforce_q_limits=false, max_outer_iter=10, q_tol=1e-6)

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
- `enforce_q_limits=false`: Whether to enforce generator reactive power limits
- `max_outer_iter=10`: Maximum Q limit enforcement iterations (if `enforce_q_limits=true`)
- `q_tol=1e-6`: Tolerance for Q limit violation detection (if `enforce_q_limits=true`)

# Returns
- `PowerFlowProblem`: The solved power flow problem with convergence information

# Examples
```julia
# Solve a single scenario power flow
prob = run_pf("case9.m")

# Solve with custom tolerance and verbosity
prob = run_pf("case9.m", CPU(), :polar; rtol=1e-10, verbose=1)

# Solve with Q limit enforcement
prob = run_pf("case9.m"; enforce_q_limits=true)

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
    max_iter=20, verbose=0,
    enforce_q_limits::Bool=false,
    max_outer_iter::Int=10,
    q_tol::Float64=1e-6,
)
    # Use Q-limited power flow if requested
    if enforce_q_limits
        if formulation == :polar
            return run_pf_with_qlim(
                datafile, backend;
                rtol=rtol, max_iter=max_iter, verbose=verbose,
                max_outer_iter=max_outer_iter, q_tol=q_tol
            )
        else
            @warn "Q limit enforcement for :block_polar formulation not yet implemented. Running standard power flow."
        end
    end

    # Standard power flow
    prob = PowerFlowProblem(
        datafile, backend, formulation, nscen,
        ploads, qloads;
        rtol=rtol, max_iter=max_iter, verbose=verbose,
    )
    prob.conv = nlsolve!(prob.non_linear_solver, prob.jac, prob.stack; linear_solver=prob.linear_solver)
    return prob
end

"""
    solve!(prob::PowerFlowProblem; enforce_q_limits=false)

Re-solve an existing power flow problem.

This function re-runs the non-linear solver on the problem, which is useful after
modifying problem parameters (e.g., via `set_active_load!` or `set_reactive_load!`).

# Arguments
- `prob::PowerFlowProblem`: The power flow problem to solve

# Keyword Arguments
- `enforce_q_limits::Bool=false`: Whether to enforce generator Q limits.
  Note: Q limit enforcement for `solve!` is not yet fully supported.
  Use `run_pf(...; enforce_q_limits=true)` for full Q limit enforcement.

# Returns
- Convergence status of the solver

# Examples
```julia
# Create and solve initial problem
prob = PowerFlowProblem("case9.m", CPU(), :polar)
solve!(prob)

# Modify parameters and re-solve
set_active_load!(prob, ones(9) * 0.8)
solve!(prob)
```

# See also
[`run_pf`](@ref), [`set_active_load!`](@ref), [`set_reactive_load!`](@ref)
"""
function solve!(prob::PowerFlowProblem; enforce_q_limits::Bool=false)
    if enforce_q_limits
        @warn "Q limit enforcement for solve!() is not yet fully supported. " *
              "Use run_pf(...; enforce_q_limits=true) for full Q limit enforcement."
    end
    prob.conv = nlsolve!(prob.non_linear_solver, prob.jac, prob.stack; linear_solver=prob.linear_solver)
    return prob.conv
end

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

#=============================================================================
    Q Limit Enforcement Functions
=============================================================================#

"""
    run_pf_with_qlim(datafile, backend=CPU(); rtol=1e-8, max_iter=20,
                     max_outer_iter=10, verbose=0, q_tol=1e-6)

Run power flow with reactive power limit enforcement.

# Arguments
- `datafile::String`: Path to network data file
- `backend::KA.Backend=CPU()`: Computation backend

# Keyword Arguments
- `rtol=1e-8`: Newton-Raphson convergence tolerance
- `max_iter=20`: Maximum Newton-Raphson iterations per solve
- `max_outer_iter=10`: Maximum Q limit enforcement iterations
- `verbose=0`: Verbosity level
- `q_tol=1e-6`: Tolerance for Q limit violation detection

# Returns
- `PowerFlowProblem`: Solved power flow problem with Q limit results
"""
function run_pf_with_qlim(
    datafile::String,
    backend::KA.Backend=CPU();
    rtol::Float64=1e-8,
    max_iter::Int=20,
    max_outer_iter::Int=10,
    verbose::Int=0,
    q_tol::Float64=1e-6
)
    network = PS.PowerNetwork(datafile)
    return run_pf_with_qlim(network, backend; rtol, max_iter, max_outer_iter, verbose, q_tol)
end

"""
    run_pf_with_qlim(network::PS.PowerNetwork, backend=CPU(); kwargs...)

Run power flow with Q limit enforcement using an existing PowerNetwork.

See `run_pf_with_qlim(datafile::String, ...)` for full documentation.
"""
function run_pf_with_qlim(
    original_network::PS.PowerNetwork,
    backend::KA.Backend=CPU();
    rtol::Float64=1e-8,
    max_iter::Int=20,
    max_outer_iter::Int=10,
    verbose::Int=0,
    q_tol::Float64=1e-6
)
    current_network = original_network

    # Track modifications
    pv_to_pq = Int[]
    q_fixed = Dict{Int, Float64}()
    all_violations = QLimitStatus[]
    total_iterations = 0
    current_bustype = copy(original_network.bustype)

    for outer_iter in 1:max_outer_iter
        # Create formulation and solve
        polar = PolarForm(current_network, backend)
        stack = NetworkStack(polar)
        powerflow = PowerFlowBalance(polar) ∘ Basis(polar)
        mapx = mapping(polar, State())
        jac = Jacobian(polar, powerflow, mapx)
        set_params!(jac, stack)

        solver = NewtonRaphson(; maxiter=max_iter, tol=rtol, verbose=verbose)
        conv = nlsolve!(solver, jac, stack)
        total_iterations += conv.n_iterations

        if !conv.has_converged
            verbose >= 1 && println("Power flow did not converge at outer iteration $outer_iter")
            qgen = compute_generator_reactive_power(polar, stack)
            return _create_qlim_problem(
                current_network, backend, polar, stack, jac, solver, conv,
                QLimitEnforcementResult(false, outer_iter, total_iterations, all_violations, qgen),
                rtol, max_iter, verbose
            )
        end

        # Check for Q limit violations
        violations = check_q_violations(polar, stack, current_bustype; tol=q_tol)

        if isempty(violations)
            # No violations - converged successfully
            verbose >= 1 && println("Q-limit enforcement converged after $outer_iter iteration(s)")
            qgen = compute_generator_reactive_power(polar, stack)
            return _create_qlim_problem(
                current_network, backend, polar, stack, jac, solver, conv,
                QLimitEnforcementResult(true, outer_iter, total_iterations, all_violations, qgen),
                rtol, max_iter, verbose
            )
        end

        verbose >= 1 && println("Q-limit iteration $outer_iter: $(length(violations)) violation(s)")

        # Process violations
        for v in violations
            push!(all_violations, v)
            bus = v.bus_idx

            # Skip reference bus
            if current_bustype[bus] == PS.REF_BUS_TYPE
                @warn "Generator $(v.gen_idx) at reference bus $bus hit Q limit - cannot convert"
                continue
            end

            if !(bus in pv_to_pq)
                push!(pv_to_pq, bus)
                current_bustype[bus] = PS.PQ_BUS_TYPE
            end

            # Aggregate Q for buses with multiple generators
            q_at_bus = get(q_fixed, bus, 0.0)
            q_fixed[bus] = q_at_bus + v.q_limit
        end

        # Rebuild network with modified bus types
        current_network = modify_bus_types(original_network, pv_to_pq, q_fixed)
    end

    # Max iterations reached without full convergence
    verbose >= 1 && println("Q-limit enforcement reached max iterations ($max_outer_iter)")
    polar = PolarForm(current_network, backend)
    stack = NetworkStack(polar)
    powerflow = PowerFlowBalance(polar) ∘ Basis(polar)
    mapx = mapping(polar, State())
    jac = Jacobian(polar, powerflow, mapx)
    set_params!(jac, stack)
    solver = NewtonRaphson(; maxiter=max_iter, tol=rtol, verbose=verbose)
    conv = nlsolve!(solver, jac, stack)

    qgen = compute_generator_reactive_power(polar, stack)
    return _create_qlim_problem(
        current_network, backend, polar, stack, jac, solver, conv,
        QLimitEnforcementResult(false, max_outer_iter, total_iterations, all_violations, qgen),
        rtol, max_iter, verbose
    )
end

"""
Helper function to create a PowerFlowProblem with Q limit results.
"""
function _create_qlim_problem(
    network, backend, polar, stack, jac, solver, conv, qlim_result,
    rtol, max_iter, verbose
)
    powerflow = PowerFlowBalance(polar) ∘ Basis(polar)
    mapx = mapping(polar, State())

    # Create linear solver
    linear_solver = LS.default_linear_solver(jac.J)

    # PowerFlowProblem struct includes qlim_result field
    return PowerFlowProblem(
        polar, stack, powerflow, linear_solver, solver, mapx, jac, conv, backend, qlim_result
    )
end

#=============================================================================
    Q Limit Accessor Functions
=============================================================================#

"""
    get_reactive_power_limits(prob::PowerFlowProblem) -> (q_min, q_max)

Get the reactive power limits for all generators.

# Returns
- Tuple of vectors `(q_min, q_max)` in per-unit
"""
function get_reactive_power_limits(prob::PowerFlowProblem)
    return PS.bounds(prob.form.network, PS.Generators(), PS.ReactivePower())
end

"""
    get_generator_reactive_power(prob::PowerFlowProblem) -> Vector{Float64}

Get the current reactive power output for all generators after power flow solution.

# Returns
- Vector of Q values in per-unit
"""
function get_generator_reactive_power(prob::PowerFlowProblem)
    return compute_generator_reactive_power(prob.form, prob.stack)
end

"""
    get_qlimit_result(prob::PowerFlowProblem) -> Union{QLimitEnforcementResult, Nothing}

Get the Q limit enforcement result from a solved power flow problem.

# Returns
- `QLimitEnforcementResult` if Q limits were enforced, `nothing` otherwise
"""
function get_qlimit_result(prob::PowerFlowProblem)
    return prob.qlim_result
end

"""
    get_violated_generators(prob::PowerFlowProblem) -> Vector{QLimitStatus}

Get list of generators that violated their Q limits during power flow.

# Returns
- Vector of `QLimitStatus` (empty if no violations or Q limits not enforced)
"""
function get_violated_generators(prob::PowerFlowProblem)
    result = get_qlimit_result(prob)
    return isnothing(result) ? QLimitStatus[] : result.violated_generators
end

"""
    is_qlimit_converged(prob::PowerFlowProblem) -> Bool

Check if power flow with Q limit enforcement converged.

# Returns
- `true` if converged or if Q limits were not enforced (standard PF)
"""
function is_qlimit_converged(prob::PowerFlowProblem)
    result = get_qlimit_result(prob)
    return isnothing(result) ? true : result.converged
end

"""
    get_bus_reactive_power(prob::PowerFlowProblem) -> Vector{Float64}

Get the reactive power injection at each bus after power flow solution.

# Returns
- Vector of Q values in per-unit for all buses
"""
function get_bus_reactive_power(prob::PowerFlowProblem)
    return compute_bus_reactive_power(prob.form, prob.stack)
end

"""
    get_generators_at_limit(prob::PowerFlowProblem; tol=1e-6) -> (at_qmin, at_qmax)

Get indices of generators currently at their Q limits.

# Returns
- Tuple of vectors `(generators at Qmin, generators at Qmax)`
"""
function get_generators_at_limit(prob::PowerFlowProblem; tol::Float64=1e-6)
    qgen = get_generator_reactive_power(prob)
    q_min, q_max = get_reactive_power_limits(prob)

    at_qmin = findall(i -> abs(qgen[i] - q_min[i]) < tol, 1:length(qgen))
    at_qmax = findall(i -> abs(qgen[i] - q_max[i]) < tol, 1:length(qgen))

    return (at_qmin, at_qmax)
end

end
