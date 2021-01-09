
# AutoDiff Factory
abstract type AbstractAutoDiffFactory end

struct AutoDiffFactory <: AbstractAutoDiffFactory
    Jgₓ::AutoDiff.Jacobian
    Jgᵤ::AutoDiff.Jacobian
    ∇f::AdjointStackObjective
end

function transfer!(target::AutoDiffFactory, origin::AutoDiffFactory)
    AutoDiff.transfer!(target.Jgₓ, origin.Jgₓ)
    AutoDiff.transfer!(target.Jgᵤ, origin.Jgᵤ)
    transfer!(target.∇f, origin.∇f)
end

# Counters
abstract type AbstractCounter end

mutable struct NLPCounter <: AbstractCounter
    objective::Int
    gradient::Int
    hessian::Int
    jacobian::Int
    jtprod::Int
    hprod::Int
end
NLPCounter() = NLPCounter(0, 0, 0, 0, 0, 0)

function Base.empty!(c::NLPCounter)
    for attr in fieldnames(NLPCounter)
        setfield!(c, attr, 0)
    end
end

# Active set utils
function _check(val, val_min, val_max)
    violated_inf = findall(val .< val_min)
    violated_sup = findall(val .> val_max)
    n_inf = length(violated_inf)
    n_sup = length(violated_sup)
    err_inf = norm(val_min[violated_inf] .- val[violated_inf], Inf)
    err_sup = norm(val[violated_sup] .- val_max[violated_sup] , Inf)
    return (n_inf, err_inf, n_sup, err_sup)
end

function _inf_pr(nlp::AbstractNLPEvaluator, cons)
    (n_inf, err_inf, n_sup, err_sup) = _check(cons, nlp.g_min, nlp.g_max)
    return max(err_inf, err_sup)
end

function active_set(c, c♭, c♯; tol=1e-8)
    @assert length(c) == length(c♭) == length(c♯)
    active_lb = findall(c .< c♭ .+ tol)
    active_ub = findall(c .> c♯ .- tol)
    return active_lb, active_ub
end

# Scaler utils
abstract type AbstractScaler end

scale_factor(h, tol, η) = max(tol, η / max(1.0, h))

struct MaxScaler{T} <: AbstractScaler
    scale_obj::T
    scale_cons::AbstractVector{T}
    g_min::AbstractVector{T}
    g_max::AbstractVector{T}
end
function MaxScaler(g_min, g_max)
    @assert length(g_min) == length(g_max)
    m = length(g_min)
    sc = similar(g_min) ; fill!(sc, 1.0)
    return MaxScaler(1.0, sc, g_min, g_max)
end
function MaxScaler(nlp::AbstractNLPEvaluator, u0::AbstractVector;
                   η=100.0, tol=1e-8)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    conv = update!(nlp, u0)
    ∇g = similar(u0) ; fill!(∇g, 0)
    gradient!(nlp, ∇g, u0)
    s_obj = scale_factor(norm(∇g, Inf), tol, η)

    # TODO: avoid forming whole Jacobian
    jac = similar(u0, (m, n)) ; fill!(jac, 0)
    s_cons = similar(u0, m) ; fill!(s_cons, 0)
    jacobian!(nlp, jac, u0)
    for i in eachindex(s_cons)
        ∇c = @view jac[i, :]
        s_cons[i] = scale_factor(norm(∇c, Inf), tol, η)
    end

    g♭, g♯ = bounds(nlp, Constraints())

    return MaxScaler(s_obj, s_cons, s_cons .* g♭, s_cons .* g♯)
end

