
abstract type AbstractPenalty end

Base.size(penal::AbstractPenalty) = length(penal.coefs)

struct QuadraticPenalty{T} <: AbstractPenalty
    coefs::AbstractVector{T}
    η::T
    c♯::T
end
function QuadraticPenalty(nlp::AbstractNLPEvaluator, cons::Function)
    if !is_constraint(cons)
        error("Function $cons is not a valid constraint function")
    end
    n = size_constraint(nlp.model, cons)
    # Default coefficients
    coefs = similar(nlp.x, n)
    fill!(coefs, 0.0)
    η = 1.0
    c♯ = 1e4
    return QuadraticPenalty(coefs, η, c♯)
end
function (penalty::QuadraticPenalty)(cx::AbstractVector)
    return 0.5 * dot(penalty.coefs, cx .* cx)
end
function update!(penal::QuadraticPenalty, cx::AbstractVector)
    penal.coefs .= min.(penal.η .* cx, penal.c♯)
end
function gradient!(
    nlp::AbstractNLPEvaluator,
    grad::AbstractVector,
    penal::QuadraticPenalty,
    cons::Function,
    u::AbstractVector,
    cx::AbstractVector
)
    jtprod!(nlp, cons, grad, u, cx)
end

struct AugLagPenalty{T} <: AbstractPenalty
    coefs::AbstractVector{T}
    μ::AbstractVector{T}
end

# Ref: https://github.com/JuliaSmoothOptimizers/Percival.jl/blob/master/src/AugLagModel.jl
mutable struct PenaltyEvaluator{T} <: AbstractNLPEvaluator
    nlp::AbstractNLPEvaluator
    cons::AbstractVector{T}
    infeasibility::AbstractVector{T}
    penalties::Vector{AbstractPenalty}
end
function PenaltyEvaluator(nlp::ReducedSpaceEvaluator;
                          penalties=AbstractPenalty[])

    if n_constraints(nlp) == 0
        @warn("Original model has no inequality constraint")
    end
    if length(penalties) != nlp.constraints
        penalties = AbstractPenalty[QuadraticPenalty(nlp, cons) for cons in nlp.constraints]
    end

    cons = similar(nlp.g_min)
    cx = similar(nlp.g_min)

    return PenaltyEvaluator(nlp, cons, cx, penalties)
end

function update!(pen::PenaltyEvaluator, u)
    update!(pen.nlp, u)
    # Update constraints
    constraint!(pen.nlp, pen.cons, u)
    # Update infeasibility error
    g♭ = pen.nlp.g_min
    g♯ = pen.nlp.g_max
    pen.infeasibility .= max.(0, pen.cons .- g♯) + min.(0, pen.cons .- g♭)
end

function update_penalty!(pen::PenaltyEvaluator)
    fr_ = 0
    for penalty in pen.penalties
        n = size(penalty)
        mask = fr_+1:fr_+n
        update!(penalty, @view pen.infeasibility[mask])
        fr_ += n
    end
end

function objective(pen::PenaltyEvaluator, u)
    # Internal objective
    obj = objective(pen.nlp, u)
    # Add penalty terms
    fr_ = 0
    for penalty in pen.penalties
        n = size(penalty)
        mask = fr_+1:fr_+n
        obj += penalty(@view pen.infeasibility[mask])
        fr_ += n
    end
    return obj
end

function gradient!(pen::PenaltyEvaluator, grad, u)
    base_nlp = pen.nlp
    gradient!(base_nlp, grad, u)
    fr_ = 0
    for (penalty, cons) in zip(pen.penalties, base_nlp.constraints)
        n = size(penalty)
        mask = fr_+1:fr_+n
        cx = @view pen.infeasibility[mask]
        gradient!(base_nlp, grad, penalty, cons, u, cx)
        fr_ += n
    end
end

