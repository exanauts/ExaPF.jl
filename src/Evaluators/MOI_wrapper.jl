# MOI wrapper

"""
    MOIEvaluator <: MOI.AbstractNLPEvaluator

Bridge from a [`ExaPF.AbstractNLPEvaluator`](@ref) to a `MOI.AbstractNLPEvaluator`.

## Attributes

* `nlp::AbstractNLPEvaluator`: the underlying `ExaPF` problem.
* `hash_x::UInt`: hash of the last evaluated variable `x`
* `has_hess::Bool` (default: `false`): if `true`, pass a Hessian structure to MOI.

"""
mutable struct MOIEvaluator{Evaluator<:AbstractNLPEvaluator,Hess} <: MOI.AbstractNLPEvaluator
    nlp::Evaluator
    hash_x::UInt
    has_hess::Bool
    hess_buffer::Hess
end
# MOI needs Hessian of Lagrangian function
function MOIEvaluator(nlp)
    hess = nothing
    if has_hessian_lagrangian(nlp)
        n = n_variables(nlp)
        hess = zeros(n, n)
    end
    return MOIEvaluator(nlp, UInt64(0), has_hessian_lagrangian(nlp), hess)
end

function _update!(ev::MOIEvaluator, x)
    hx = hash(x)
    if hx != ev.hash_x
        update!(ev.nlp, x)
        ev.hash_x = hx
    end
end

MOI.features_available(ev::MOIEvaluator) = ev.has_hess ? [:Grad, :Hess] : [:Grad]
MOI.initialize(ev::MOIEvaluator, features) = nothing

function MOI.jacobian_structure(ev::MOIEvaluator)
    rows, cols = jacobian_structure(ev.nlp)
    return Tuple{Int, Int}[(r, c) for (r, c) in zip(rows, cols)]
end

function MOI.hessian_lagrangian_structure(ev::MOIEvaluator)
    n = n_variables(ev.nlp)
    rows, cols = hessian_structure(ev.nlp)
    return Tuple{Int, Int}[(r, c) for (r, c) in zip(rows, cols)]
end

function MOI.eval_objective(ev::MOIEvaluator, x)
    _update!(ev, x)
    obj = objective(ev.nlp, x)
    return obj
end

function MOI.eval_objective_gradient(ev::MOIEvaluator, g, x)
    _update!(ev, x)
    gradient!(ev.nlp, g, x)
end

function MOI.eval_constraint(ev::MOIEvaluator, cons, x)
    _update!(ev, x)
    constraint!(ev.nlp, cons, x)
end

function MOI.eval_constraint_jacobian(ev::MOIEvaluator, jac, x)
    n = length(x)
    m = n_constraints(ev.nlp)
    _update!(ev, x)
    fill!(jac, 0)
    J = reshape(jac, m, n)
    jacobian!(ev.nlp, J, x)
end

function MOI.eval_hessian_lagrangian(ev::MOIEvaluator, hess, x, σ, μ)
    _update!(ev, x)
    n = n_variables(ev.nlp)
    # Evaluate full reduced Hessian in the preallocated buffer.
    H = ev.hess_buffer
    hessian!(ev.nlp, H, x)
    # Only dense Hessian supported now
    index = 1
    @inbounds for i in 1:n, j in 1:i
        # Hessian is symmetric, and MOI considers only the lower
        # triangular part. We average the values from the lower
        # and upper triangles for stability.
        hess[index] = 0.5 * σ * (H[i, j] + H[j, i])
        index += 1
    end
end

function MOI.eval_hessian_lagrangian_product(ev::MOIEvaluator, hv, x, v, σ, μ)
    _update!(ev, x)
    hessprod!(ev.nlp, hv, x, v)
    hv .*= σ
end

function MOI.NLPBlockData(nlp::AbstractNLPEvaluator)
    lb, ub = bounds(nlp, Constraints())
    ev = MOIEvaluator(nlp)
    return MOI.NLPBlockData(MOI.NLPBoundsPair.(lb, ub), ev, true)
end

