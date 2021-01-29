# Code for augmented Lagrangian evaluator. Inspired by the excellent NLPModels.jl:
# Ref: https://github.com/JuliaSmoothOptimizers/Percival.jl/blob/master/src/AugLagModel.jl
# Two-sided Lagrangian

const CONSTRAINTS_TYPE = Union{Val{:inequality}, Val{:equality}, Val{:mixed}}

@doc raw"""
    AugLagEvaluator{T} <: AbstractPenaltyEvaluator

Augmented-Lagrangian evaluator. Takes as input any `AbstractNLPEvaluator`
encoding a non-linear problem
```math
\begin{aligned}
       \min_u \quad & f(u)\\
\mathrm{s.t.} \quad & h♭ ≤ h(u) ≤ h♯,\\
                    & u♭ ≤  u   ≤ u♯,
\end{aligned}
```
and return a new evaluator reformulating the original problem
by moving the `m` constraints `h♭ ≤ h(u) ≤ h♯` into the objective
using a set of penalties `ϕ_1, ⋯, ϕ_m`:
```math
\begin{aligned}
    \min_u \quad & f(u) + \sum_i ϕ_i(h_i)   \\
\mathrm{s.t.} \quad &  u♭ ≤  u   ≤  u♯,
\end{aligned}
```

This evaluator considers explicitly the inequality constraints,
without reformulating them by introducing slack variables. Each
penalty `ϕ_i` writes
```math
ϕ_i(h_i) = λ_i' * φ(h_i) + \frac \rho2 \| φ(h_i) \|_2^2
```
with `φ` a function to compute the current infeasibility (hence equal to 0
if `h_i` is feasible)
```math
φ(h_i) = \max\{0 , λ_i + ρ * (h_i - h_i♯)   \} + \min\{0 , λ_i + ρ * (h_i - h_i♭)   \}
```
"""
mutable struct AugLagEvaluator{T} <: AbstractPenaltyEvaluator
    inner::AbstractNLPEvaluator
    # Type
    cons_type::CONSTRAINTS_TYPE
    cons::AbstractVector{T}
    ρ::T
    λ::AbstractVector{T}
    λc::AbstractVector{T}
    # Scaling
    scaler::AbstractScaler
    # Stats
    counter::AbstractCounter
end
function AugLagEvaluator(
    nlp::AbstractNLPEvaluator, u0;
    scale=false, c₀=0.1,
)
    if !is_constrained(nlp)
        error("Model specified in `nlp` is unconstrained. Instead of using" *
              " an Augmented Lagrangian algorithm, you could use any "*
              "bound constrained solver instead.")
    end
    cons_type = Val(constraints_type(nlp))
    if !isa(cons_type, CONSTRAINTS_TYPE)
        error("Constraints $(constraints_type(nlp)) is not supported by" *
              " AugLagEvaluator.")
    end

    g_min, g_max = bounds(nlp, Constraints())
    cx = similar(g_min) ; fill!(cx, 0)
    λc = similar(g_min) ; fill!(λc, 0)
    λ = similar(g_min) ; fill!(λ, 0)


    scaler = scale ?  MaxScaler(nlp, u0) : MaxScaler(g_min, g_max)
    return AugLagEvaluator(nlp, cons_type, cx, c₀, λ, λc, scaler, NLPCounter())
end
function AugLagEvaluator(
    datafile::String; options...
)
    nlp = ReducedSpaceEvaluator(datafile)
    u0 = initial(nlp)
    return AugLagEvaluator(nlp, u0; options...)
end

# Default fallback
function _update_internal!(ag::AugLagEvaluator, ::CONSTRAINTS_TYPE)
    # Update (shifted) infeasibility error
    g♭ = ag.scaler.g_min
    g♯ = ag.scaler.g_max
    ag.λc .= max.(0, ag.λ .+ ag.ρ .* (ag.cons .- g♯)) .+
             min.(0, ag.λ .+ ag.ρ .* (ag.cons .- g♭))
    ag.cons .= max.(0, ag.cons .- g♯) .+ min.(0, ag.cons .- g♭)
end

# Specialization for equality constraints
function _update_internal!(ag::AugLagEvaluator, ::Val{:equality})
    ag.λc .= ag.λ .+ ag.ρ .* ag.cons
end

function update!(ag::AugLagEvaluator, u)
    conv = update!(ag.inner, u)
    # Update constraints
    constraint!(ag.inner, ag.cons, u)
    # Rescale
    ag.cons .*= ag.scaler.scale_cons
    _update_internal!(ag, ag.cons_type)
    return conv
end

function update_penalty!(ag::AugLagEvaluator; η=10.0)
    ag.ρ = min(η * ag.ρ, 10e12)
end

function update_multipliers!(ag::AugLagEvaluator)
    ag.λ .= ag.λc
    return
end

function objective(ag::AugLagEvaluator, u)
    ag.counter.objective += 1
    base_nlp = ag.inner
    cx = ag.cons
    obj = ag.scaler.scale_obj * objective(base_nlp, u) +
        0.5 * ag.ρ * dot(cx, cx) + dot(ag.λ, cx)
    return obj
end

function inner_objective(ag::AugLagEvaluator, u)
    return ag.scaler.scale_obj * objective(ag.inner, u)
end

function gradient!(ag::AugLagEvaluator, grad, u)
    ag.counter.gradient += 1
    base_nlp = ag.inner
    scaler = ag.scaler
    σ = scaler.scale_obj
    mask = abs.(ag.cons) .> 0
    v = scaler.scale_cons .* ag.λc .* mask
    ojtprod!(base_nlp, grad, u, σ, v)
end

# TODO: currently, hessprod! works only with the `ReducedSpaceEvaluator`
# We should implement something equivalent to `ojtprod!`, but for computing
# second order information.
function hessprod!(ag::AugLagEvaluator, hessvec, u, w)
    base_nlp = ag.inner
    scaler = ag.scaler
    # Import AutoDiff objects
    autodiff = get(base_nlp, AutoDiffBackend())
    cx = ag.cons
    mask = abs.(cx) .> 1e-10

    σ = scaler.scale_obj
    y = scaler.scale_cons .* ag.λc .* mask
    # Hessian of Lagrangian L(u, y) = f(u) + y' * h(u)
    ∇²L = full_hessian_lagrangian(base_nlp, u, y, σ)
    J = full_jacobian(base_nlp, u)
    D = Diagonal(scaler.scale_cons .* scaler.scale_cons .* mask)
    ∇²L.xx .+= ag.ρ .* J.x' * D * J.x
    ∇²L.xu .+= ag.ρ .* J.u' * D * J.x
    ∇²L.uu .+= ag.ρ .* J.u' * D * J.u

    reduced_hessian!(base_nlp, hessvec, ∇²L.xx, ∇²L.xu, ∇²L.uu, w)
end

function reset!(ag::AugLagEvaluator)
    reset!(ag.inner)
    empty!(ag.counter)
    fill!(ag.cons, 0)
    fill!(ag.λ, 0)
    fill!(ag.λc, 0)
end

