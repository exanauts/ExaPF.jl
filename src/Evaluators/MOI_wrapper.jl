# MOI wrapper

"""
    MOIEvaluator <: MOI.AbstractNLPEvaluator

Bridge from a `ExaPF.AbstractNLPEvaluator` to a `MOI.AbstractNLPEvaluator`.

## Attributes

* `nlp::AbstractNLPEvaluator`: the underlying `ExaPF` problem.
* `hash_x::UInt`: hash of the last evaluated variable `x`
* `has_hess::Bool` (default: `false`): if `true`, pass a Hessian structure to MOI.

"""
mutable struct MOIEvaluator <: MOI.AbstractNLPEvaluator
    nlp::AbstractNLPEvaluator
    hash_x::UInt
    has_hess::Bool
end
MOIEvaluator(nlp) = MOIEvaluator(nlp, UInt64(0), true)

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

function MOI.eval_constraint_jacobian(ev::MOIEvaluator, ∂cons, x)
    _update!(ev, x)
    n = n_variables(ev.nlp)
    m = n_constraints(ev.nlp)
    # Build up a dense Jacobian
    jac = zeros(m, n)
    jacobian!(ev.nlp, jac, x)
    # Copy back to the MOI arrray
    rows, cols = jacobian_structure(ev.nlp)
    k = 1
    for (i, j) in zip(rows, cols)
        ∂cons[k] = jac[i, j]
        k += 1
    end
end

function MOI.eval_hessian_lagrangian(ev::MOIEvaluator, H, x, σ, μ)
    _update!(ev, x)
    n = n_variables(ev.nlp)
    hess = zeros(n, n)
    hessian_lagrangian!(ev.nlp, hess, x, μ, σ)
    # Copy back to the MOI arrray
    rows, cols = hessian_structure(ev.nlp)
    k = 1
    for (i, j) in zip(rows, cols)
        H[k] = hess[i, j]
        k += 1
    end
end

function MOI.NLPBlockData(nlp::AbstractNLPEvaluator)
    lb, ub = bounds(nlp, Constraints())
    ev = MOIEvaluator(nlp)
    return MOI.NLPBlockData(MOI.NLPBoundsPair.(lb, ub), ev, true)
end

