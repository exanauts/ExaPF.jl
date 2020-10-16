
scale_factor(h, tol) = max(tol, 100.0 / max(1.0, h))

mutable struct ScalingEvaluator{T} <: AbstractNLPEvaluator
    inner::AbstractNLPEvaluator
    scale_obj::T
    scale_cons::AbstractVector{T}
    tol::T
    g_min::AbstractVector{T}
    g_max::AbstractVector{T}
end
function ScalingEvaluator(nlp::AbstractNLPEvaluator, u0::AbstractVector; tol=1e-8)
    n, m = n_variables(nlp), n_constraints(nlp)

    update!(nlp, u0)
    ∇g = similar(u0, n) ; fill!(∇g, 0)
    gradient!(nlp, ∇g, u0)
    s_obj = scale_factor(norm(∇g, Inf), tol)

    # TODO: avoid forming whole Jacobian
    jac = similar(u0, (m, n))
    s_cons = similar(u0, m)
    jacobian!(nlp, jac, u0)
    for i in eachindex(s_cons)
        ∇c = @view jac[i, :]
        s_cons[i] = scale_factor(norm(∇c, Inf), tol)
    end
    return ScalingEvaluator{Float64}(nlp, s_obj, s_cons, tol, s_cons .* nlp.g_min, s_cons .* nlp.g_max)
end

n_variables(ev::ScalingEvaluator) = length(ev.u_min)
n_constraints(ev::ScalingEvaluator) = length(ev.g_min)

# Inherit attributes of underlying evaluator
function Base.getproperty(ev::ScalingEvaluator, name::Symbol)
    if !hasfield(typeof(ev), name)
        inner_model = getfield(ev, :inner)
        return getfield(inner_model, name)
    else
        return getfield(ev, name)
    end
end

update!(ev::ScalingEvaluator, u) = update!(ev.inner, u)

objective(ev::ScalingEvaluator, u) = ev.scale_obj * objective(ev.inner, u)

function gradient!(ev::ScalingEvaluator, grad, u)
    gradient!(ev.inner, grad, u)
    grad .*= ev.scale_obj
    return
end

function constraint!(ev::ScalingEvaluator, cons, u)
    constraint!(ev.inner, cons, u)
    cons .*= ev.scale_cons
end

jacobian_structure!(ev::ScalingEvaluator, rows, cols) = jacobian_structure!(ev.inner, rows, cols)

function jacobian!(ev::ScalingEvaluator, jac, u)
    jacobian!(ev.inner, jac, u)
    jac .= Diagonal(ev.scale_cons) * jac
end

function jtprod!(ev::ScalingEvaluator, cons, jv, u, v; shift=1, start=1)
    fr_, to_ = start, start + size_constraint(ev.model, cons) - 1
    w = v .* ev.scale_cons[fr_:to_]
    jtprod!(ev.inner, cons, jv, u, w; shift=shift)
end
function jtprod!(ev::ScalingEvaluator, jv, u, v)
    w = v .* ev.scale_cons
    jtprod!(ev.inner, jv, u, w)
end

