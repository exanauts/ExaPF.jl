# MOI wrapper

mutable struct MOIEvaluator <: MOI.AbstractNLPEvaluator
    nlp::AbstractNLPEvaluator
    hash_x::UInt
    has_hess::Bool
end
MOIEvaluator(nlp) = MOIEvaluator(nlp, UInt64(0), false)

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
    n = n_variables(ev.nlp)
    m = n_constraints(ev.nlp)
    jnnz = n * m
    rows = zeros(Int, jnnz)
    cols = zeros(Int, jnnz)
    jacobian_structure!(ev.nlp, rows, cols)
    return Tuple{Int, Int}[(r, c) for (r, c) in zip(rows, cols)]
end

function MOI.hessian_lagrangian_structure(ev::MOIEvaluator)
    n = n_variables(ev.nlp)
    return Tuple{Int, Int}[(i, i) for i in 1:n]
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
    jac = zeros(m, n)
    jacobian!(ev.nlp, jac, x)
    k = 1
    for i = 1:m
        for j = 1:n
            ∂cons[k] = jac[i, j]
            k += 1
        end
    end
end

function MOI.eval_hessian_lagrangian(ev::MOIEvaluator, H, x, σ, μ)
    fill!(H, 1.0)
end

function MOI.NLPBlockData(nlp::AbstractNLPEvaluator)
    lb, ub = bounds(nlp, Constraints())
    ev = MOIEvaluator(nlp)
    return MOI.NLPBlockData(MOI.NLPBoundsPair.(lb, ub), ev, true)
end

