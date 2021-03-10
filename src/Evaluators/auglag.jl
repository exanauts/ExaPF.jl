# Code for augmented Lagrangian evaluator. Inspired by the excellent NLPModels.jl:
# Ref: https://github.com/JuliaSmoothOptimizers/Percival.jl/blob/master/src/AugLagModel.jl
# Two-sided Lagrangian

const CONSTRAINTS_TYPE = Union{Val{:inequality}, Val{:equality}, Val{:mixed}}

@doc raw"""
    AugLagEvaluator{Evaluator<:AbstractNLPEvaluator, T, VT} <: AbstractPenaltyEvaluator

Augmented-Lagrangian evaluator.

### Description

Takes as input any `AbstractNLPEvaluator` encoding a non-linear problem
```math
\begin{aligned}
       \min_u \quad & f(u)\\
\mathrm{s.t.} \quad & h♭ ≤ h(u) ≤ h♯,\\
                    & u♭ ≤  u   ≤ u♯,
\end{aligned}
```
and return a new evaluator reformulating the original problem
by moving the $m$ constraints $h♭ ≤ h(u) ≤ h♯$ into the objective
using a set of penalties $ϕ_1, ⋯, ϕ_m$ and multiplier estimates
$λ_1, ⋯, λ_m$:
```math
\begin{aligned}
    \min_u \quad & f(u) + \sum_{i=1}^m ϕ_i(h_i, λ_i)   \\
\mathrm{s.t.} \quad &  u♭ ≤  u   ≤  u♯,
\end{aligned}
```

This evaluator considers explicitly the inequality constraints,
without reformulating them by introducing slack variables. Each
penalty $ϕ_i$ is defined as
```math
ϕ_i(h_i, λ_i) = λ_i^⊤ φ_i(h_i) + \frac \rho2 \| φ_i(h_i) \|_2^2
```
with $φ_i$ a function to compute the current infeasibility
```math
φ_i(h_i, λ_i) = \max\{0 , λ_i + ρ (h_i - h_i♯)   \} + \min\{0 , λ_i + ρ (h_i - h_i♭)   \}
```

### Attributes

* `inner::Evaluator`: original problem.
* `cons_type`: type of the constraints of the original problem (equalities or inequalities).
* `cons::VT`: a buffer storing the current evaluation of the constraints for the inner evaluator.
* `rho::T`: current penalty.
* `λ::VT`: current multiplier.
* `scaler::MaxScaler{T,VT}`: a scaler to rescale the range of the constraints in the original problem.

"""
mutable struct AugLagEvaluator{Evaluator<:AbstractNLPEvaluator, T, VT} <: AbstractPenaltyEvaluator
    inner::Evaluator
    # Type
    cons_type::CONSTRAINTS_TYPE
    cons::VT
    ρ::T
    λ::VT
    λc::VT
    # Scaling
    scaler::MaxScaler{T, VT}
    # Stats
    counter::NLPCounter
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

has_hessian(nlp::AugLagEvaluator) = has_hessian(nlp.inner)

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
    return
end

function hessprod!(ag::AugLagEvaluator, hessvec, u, w)
    ag.counter.hprod += 1
    scaler = ag.scaler
    cx = ag.cons
    mask = abs.(cx) .> 0

    σ = scaler.scale_obj
    y = (scaler.scale_cons .* ag.λc .* mask)
    z = ag.ρ .* (scaler.scale_cons .* scaler.scale_cons .* mask)

    hessian_lagrangian_penalty_prod!(ag.inner, hessvec, u, y, σ, w, z)::Nothing
    return
end

function estimate_multipliers(ag::AugLagEvaluator, u)
    J = Diagonal(ag.scaler.scale_cons) * jacobian(ag.inner, u)
    ∇f = gradient(ag.inner, u)
    ∇f .*= ag.scaler.scale_obj
    λ = - (J * J') \ (J * ∇f)
    return λ
end

function reset!(ag::AugLagEvaluator)
    reset!(ag.inner)
    empty!(ag.counter)
    fill!(ag.cons, 0)
    fill!(ag.λ, 0)
    fill!(ag.λc, 0)
end

