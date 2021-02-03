
"""
    FeasibilityEvaluator{T} <: AbstractNLPEvaluator

TODO

"""
mutable struct FeasibilityEvaluator{T} <: AbstractNLPEvaluator
    inner::AbstractNLPEvaluator
    x_min::AbstractVector{T}
    x_max::AbstractVector{T}
    cons::AbstractVector{T}
end
function FeasibilityEvaluator(nlp::AbstractNLPEvaluator)
    if !is_constrained(nlp)
        error("Input problem must have inequality constraints")
    end
    x_min, x_max = bounds(nlp.inner, Variables())
    cx = similar(x_min, n_constraints(nlp.inner))
    return FeasibilityEvaluator(nlp, x_min, x_max, cx)
end
function FeasibilityEvaluator(datafile::String)
    nlp = SlackEvaluator(datafile)
    if !is_constrained(nlp)
        error("Input problem must have inequality constraints")
    end
    x_min, x_max = bounds(nlp.inner, Variables())
    cx = similar(x_min, n_constraints(nlp.inner))
    return FeasibilityEvaluator(nlp, x_min, x_max, cx)
end

n_variables(nlp::FeasibilityEvaluator) = n_variables(nlp.inner)
n_constraints(nlp::FeasibilityEvaluator) = 0

constraints_type(::FeasibilityEvaluator) = :bound

# Getters
get(nlp::FeasibilityEvaluator, attr::AbstractNLPAttribute) = get(nlp.inner, attr)
get(nlp::FeasibilityEvaluator, attr::AbstractVariable) = get(nlp.inner, attr)
get(nlp::FeasibilityEvaluator, attr::PS.AbstractNetworkAttribute) = get(nlp.inner, attr)

# Setters
function setvalues!(nlp::FeasibilityEvaluator, attr::PS.AbstractNetworkValues, values)
    setvalues!(nlp.inner, attr, values)
end

# Bounds
bounds(nlp::FeasibilityEvaluator, ::Variables) = bounds(nlp.inner, Variables())
bounds(nlp::FeasibilityEvaluator, ::Constraints) = (Float64[], Float64[])

initial(nlp::FeasibilityEvaluator) = initial(nlp.inner)

function update!(nlp::FeasibilityEvaluator, u)
    conv = update!(nlp.inner, u)
    constraint!(nlp.inner, nlp.cons, u)
    return conv
end

# f(x) = 0.5 * || c(x) ||²
function objective(nlp::FeasibilityEvaluator, u)
    return 0.5 * dot(nlp.cons, nlp.cons)
end

function constraint!(nlp::FeasibilityEvaluator, cons, u)
    @assert length(cons) == 0
    return
end

# Gradient
# ∇f = J' * c(x)
function gradient!(nlp::FeasibilityEvaluator, grad, u)
    σ = 0.0
    ojtprod!(nlp.inner, grad, u, 0.0, nlp.cons)
end

# H = ∇²c(x) + J'*J
function hessprod!(nlp::FeasibilityEvaluator, hessvec, u, v)
    σ = 0.0
    # Need to update the first-order adjoint λ first
    hessian_lagrangian_prod!(nlp.inner, hessvec, u, nlp.cons, σ, v)
    J = jacobian(nlp.inner, u)
    hessvec .+= J' * J * v
end

function hessian_structure(nlp::FeasibilityEvaluator)
    n = n_variables(nlp)
    rows = Int[r for r in 1:n for c in 1:r]
    cols = Int[c for r in 1:n for c in 1:r]
    return rows, cols
end

function reset!(nlp::FeasibilityEvaluator)
    reset!(nlp.inner)
end

