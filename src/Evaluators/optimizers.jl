
"""
    optimize!(optimizer, nlp::AbstractNLPEvaluator, x0)

Use optimization routine implemented in `optimizer` to optimize
the optimal power flow problem specified in the evaluator `nlp`.
Initial point is specified by `x0`.

Return the solution as a named tuple, with fields
- `status::MOI.TerminationStatus`: Solver's termination status, as specified by MOI
- `minimum::Float64`: final objective
- `minimizer::AbstractVector`: final solution vector, with same ordering as the `Variables` specified in `nlp`.


    optimize!(optimizer, nlp::AbstractNLPEvaluator)

Wrap previous `optimize!` function and pass as initial guess `x0`
the initial value specified when calling `initial(nlp)`.

## Examples

```julia
nlp = ExaPF.ReducedSpaceEvaluator(datafile)
optimizer = Ipopt.Optimizer()
solution = ExaPF.optimize!(optimizer, nlp)

```

## Notes
By default, the optimization routine solves a minimization
problem.

"""
function optimize! end

# MOI-based solver
function build!(optimizer::MOI.AbstractOptimizer, nlp::AbstractNLPEvaluator)
    block_data = MOI.NLPBlockData(nlp)
    u♭, u♯ = ExaPF.bounds(nlp, ExaPF.Variables())
    n = ExaPF.n_variables(nlp)
    u = MOI.add_variables(optimizer, n)
    # Set bounds
    MOI.add_constraints(
        optimizer, u, MOI.LessThan.(u♯),
    )
    MOI.add_constraints(
        optimizer, u, MOI.GreaterThan.(u♭),
    )
    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return u
end

function optimize!(optimizer::MOI.AbstractOptimizer, nlp::AbstractNLPEvaluator, x0)
    u = build!(optimizer, nlp)
    MOI.set(optimizer, MOI.VariablePrimalStart(), u, x0)
    MOI.optimize!(optimizer)
    x_opt = MOI.get(optimizer, MOI.VariablePrimal(), u)
    solution = (
        status=MOI.get(optimizer, MOI.TerminationStatus()),
        minimum=MOI.get(optimizer, MOI.ObjectiveValue()),
        minimizer=x_opt,
    )
    return solution
end
optimize!(optimizer, nlp::AbstractNLPEvaluator) = optimize!(optimizer, nlp, initial(nlp))

