abstract type AbstractPenaltyEvaluator <: AbstractNLPEvaluator end

n_variables(ag::AbstractPenaltyEvaluator) = n_variables(ag.inner)
# All constraints moved inside the objective with penalties!
n_constraints(ag::AbstractPenaltyEvaluator) = 0

# Getters
get(ag::AbstractPenaltyEvaluator, attr::AbstractNLPAttribute) = get(ag.inner, attr)

# Initial position
initial(ag::AbstractPenaltyEvaluator) = initial(ag.inner)

# Bounds
bounds(ag::AbstractPenaltyEvaluator, ::Variables) = bounds(ag.inner, Variables())
bounds(ag::AbstractPenaltyEvaluator, ::Constraints) = Float64[], Float64[]

# based objective
objective_original(ag::AbstractPenaltyEvaluator, u) = objective(ag.inner, u)

function constraint!(ag::AbstractPenaltyEvaluator, cons, u)
    @assert length(cons) == 0
    return
end

function jacobian!(ag::AbstractPenaltyEvaluator, jac, u)
    @assert length(jac) == 0
    return
end

function jacobian_structure!(ag::AbstractPenaltyEvaluator, rows, cols)
    @assert length(rows) == length(cols) == 0
end

primal_infeasibility!(ag::AbstractPenaltyEvaluator, cons, u) = primal_infeasibility!(ag.inner, cons, u)
primal_infeasibility(ag::AbstractPenaltyEvaluator, u) = primal_infeasibility(ag.inner, u)

