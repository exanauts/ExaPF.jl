# MOI wrapper

mutable struct ExaEvaluator <: MOI.AbstractNLPEvaluator
    nlp::AbstractNLPEvaluator
    hash_x::UInt
end
ExaEvaluator(nlp) = ExaEvaluator(nlp, UInt64(0))

function _update!(ev::ExaEvaluator, x)
    hx = hash(x)
    if hx != ev.hash_x
        update!(ev.nlp, x)
        ev.hash_x = hx
    end
end

MOI.features_available(::ExaEvaluator) = [:Grad]
MOI.initialize(ev::ExaEvaluator, features) = nothing

function MOI.jacobian_structure(ev::ExaEvaluator)
    n = n_variables(ev.nlp)
    m = n_constraints(ev.nlp)
    jnnz = n * m
    rows = zeros(Int, jnnz)
    cols = zeros(Int, jnnz)
    jacobian_structure!(ev.nlp, rows, cols)
    return Tuple{Int, Int}[(r, c) for (r, c) in zip(rows, cols)]
end

function MOI.eval_objective(ev::ExaEvaluator, x)
    _update!(ev, x)
    obj = objective(ev.nlp, x)
    return obj
end

function MOI.eval_objective_gradient(ev::ExaEvaluator, g, x)
    _update!(ev, x)
    gradient!(ev.nlp, g, x)
end

function MOI.eval_constraint(ev::ExaEvaluator, cons, x)
    _update!(ev, x)
    constraint!(ev.nlp, cons, x)
end

function MOI.eval_constraint_jacobian(ev::ExaEvaluator, ∂cons, x)
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

function MOI.NLPBlockData(nlp::AbstractNLPEvaluator)
    lb, ub = bounds(nlp, Constraints())
    ev = ExaEvaluator(nlp)
    return MOI.NLPBlockData(MOI.NLPBoundsPair.(lb, ub), ev, true)
end

