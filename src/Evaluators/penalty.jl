
# Ref: https://github.com/JuliaSmoothOptimizers/Percival.jl/blob/master/src/AugLagModel.jl
mutable struct PenaltyEvaluator{T} <: AbstractNLPEvaluator
    inner::AbstractNLPEvaluator
    cons::AbstractVector{T}
    infeasibility::AbstractVector{T}
    penalties::AbstractVector{T}
    # Scaling
    scaler::AbstractScaler
end

function PenaltyEvaluator(nlp::AbstractNLPEvaluator, u0;
                          scale=false,
                          penalties=Float64[],
                          c₀=0.1)

    if n_constraints(nlp) == 0
        @warn("Original model has no inequality constraint")
    end
    if length(penalties) != length(nlp.constraints)
        penalties = Float64[c₀=c₀ for cons in nlp.constraints]
    end

    cons = similar(nlp.g_min)
    cx = similar(nlp.g_min)

    if scale
        scaler = MaxScaler(nlp, u0)
    else
        scaler = MaxScaler(nlp.g_min, nlp.g_max)
    end

    return PenaltyEvaluator(nlp, cons, cx, penalties, scaler)
end

function update!(pen::PenaltyEvaluator, u)
    conv = update!(pen.inner, u)
    # Update constraints
    constraint!(pen.inner, pen.cons, u)
    # Rescale
    pen.cons .*= pen.scaler.scale_cons
    # Update infeasibility error
    g♭ = pen.scaler.g_min
    g♯ = pen.scaler.g_max
    pen.infeasibility .= max.(0, pen.cons .- g♯) .+ min.(0, pen.cons .- g♭)
    return conv
end

function update_penalty!(pen::PenaltyEvaluator; η=10.0)
    fr_ = 0
    pen.penalties = min.(η * pen.penalties, 10e12)
end

function objective(pen::PenaltyEvaluator, u)
    base_nlp = pen.inner
    # Internal objective
    obj = pen.scaler.scale_obj * objective(base_nlp, u)
    # Add penalty terms
    fr_ = 0
    for (πp, cons) in zip(pen.penalties, base_nlp.constraints)
        n = size_constraint(base_nlp.model, cons)
        mask = fr_+1:fr_+n
        cx = pen.infeasibility[mask]
        obj += 0.5 * πp * dot(cx, cx)
        fr_ += n
    end
    return obj
end

function gradient!(pen::PenaltyEvaluator, grad, u)
    base_nlp = pen.inner
    model = base_nlp.model
    scaler = pen.scaler
    λ = base_nlp.λ
    # Import buffer (has been updated previously in update!)
    buffer = base_nlp.buffer
    # Import AD objects
    ∇gᵤ = jacobian(model, base_nlp.ad.Jgᵤ, buffer)
    ∂obj = base_nlp.ad.∇f
    jvx = ∂obj.jvₓ ; fill!(jvx, 0)
    jvu = ∂obj.jvᵤ ; fill!(jvu, 0)

    # compute gradient of objective
    ∂cost(model, ∂obj, buffer)
    jvx .+= scaler.scale_obj .* ∂obj.∇fₓ
    jvu .+= scaler.scale_obj .* ∂obj.∇fᵤ

    # compute gradient of penalties
    fr_ = 0
    for (πp, cons) in zip(pen.penalties, base_nlp.constraints)
        n = size_constraint(base_nlp.model, cons)
        mask = fr_+1:fr_+n
        cx = @view pen.infeasibility[mask]
        v = cx .* πp .* scaler.scale_cons[mask]
        jtprod(model, cons, ∂obj, buffer, v)
        jvx .+= ∂obj.∇fₓ
        jvu .+= ∂obj.∇fᵤ
        fr_ += n
    end
    # evaluate reduced gradient
    LinearSolvers.ldiv!(base_nlp.linear_solver, λ, base_nlp.∇gᵗ, jvx)
    grad .= jvu
    mul!(grad, transpose(∇gᵤ), λ, -1.0, 1.0)
end

primal_infeasibility!(pen::PenaltyEvaluator, cons, u) = primal_infeasibility!(pen.inner, cons, u)
primal_infeasibility(pen::PenaltyEvaluator, u) = primal_infeasibility(pen.inner, u)

