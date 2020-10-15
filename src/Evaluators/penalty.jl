
# Ref: https://github.com/JuliaSmoothOptimizers/Percival.jl/blob/master/src/AugLagModel.jl
mutable struct PenaltyEvaluator{T} <: AbstractNLPEvaluator
    inner::AbstractNLPEvaluator
    cons::AbstractVector{T}
    infeasibility::AbstractVector{T}
    penalties::Vector{AbstractPenalty}
end
function PenaltyEvaluator(nlp::AbstractNLPEvaluator;
                          penalties=AbstractPenalty[],
                          c₀=0.1)

    if n_constraints(nlp) == 0
        @warn("Original model has no inequality constraint")
    end
    if length(penalties) != length(nlp.constraints)
        penalties = AbstractPenalty[QuadraticPenalty(size_constraint(nlp.model, cons); c₀=c₀) for cons in nlp.constraints]
    end

    cons = similar(nlp.g_min)
    cx = similar(nlp.g_min)

    return PenaltyEvaluator(nlp, cons, cx, penalties)
end

function update!(pen::PenaltyEvaluator, u)
    update!(pen.inner, u)
    # Update constraints
    constraint!(pen.inner, pen.cons, u)
    # Update infeasibility error
    g♭ = pen.inner.g_min
    g♯ = pen.inner.g_max
    pen.infeasibility .= max.(0, pen.cons .- g♯) + min.(0, pen.cons .- g♭)
end

function update_penalty!(pen::PenaltyEvaluator; η=10.0)
    fr_ = 0
    for penalty in pen.penalties
        n = size(penalty)
        mask = fr_+1:fr_+n
        update!(penalty, view(pen.infeasibility, mask), η)
        fr_ += n
    end
end

function objective(pen::PenaltyEvaluator, u)
    # Internal objective
    obj = objective(pen.inner, u)
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
    base_nlp = pen.inner
    gradient!(base_nlp, grad, u)
    fr_ = 0
    for (penalty, cons) in zip(pen.penalties, base_nlp.constraints)
        n = size(penalty)
        mask = fr_+1:fr_+n
        cx = @view pen.infeasibility[mask]
        gradient!(base_nlp, grad, penalty, cons, u, cx; start=fr_+1)
        fr_ += n
    end
end

