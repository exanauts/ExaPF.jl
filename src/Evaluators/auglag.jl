
# Ref: https://github.com/JuliaSmoothOptimizers/Percival.jl/blob/master/src/AugLagModel.jl
# TODO:
# - check case when lb = ub for inequality constraints
# Two-sided Lagrangian
mutable struct AugLagEvaluator{T} <: AbstractNLPEvaluator
    inner::AbstractNLPEvaluator
    cons::AbstractVector{T}
    infeasibility::AbstractVector{T}
    ρ::T
    λ::AbstractVector{T}
    λc::AbstractVector{T}
    # Scaling
    scaler::AbstractScaler
    # Stats
    counter::AbstractCounter
end
function AugLagEvaluator(nlp::AbstractNLPEvaluator, u0;
                         penalties=Float64[],
                         scale=false,
                         c₀=0.1)

    if n_constraints(nlp) == 0
        @warn("Original model has no inequality constraint")
    end

    cons = similar(nlp.g_min)
    cx = similar(nlp.g_min)
    λc = similar(nlp.g_min) ; fill!(λc, 0)
    λ = similar(nlp.g_min) ; fill!(λ, 0)

    scaler = scale ?  MaxScaler(nlp, u0) : MaxScaler(nlp.g_min, nlp.g_max)
    return AugLagEvaluator(nlp, cons, cx, c₀, λ, λc, scaler, NLPCounter())
end

initial(ag::AugLagEvaluator) = initial(ag.inner)
bounds(ag::AugLagEvaluator, ::Variables) = ag.inner.u_min, ag.inner.u_max
bounds(ag::AugLagEvaluator, ::Constraints) = Float64[], Float64[]
n_variables(ag::AugLagEvaluator) = n_variables(ag.inner)
n_constraints(ag::AugLagEvaluator) = 0

function update!(ag::AugLagEvaluator, u)
    conv = update!(ag.inner, u)
    # Update constraints
    constraint!(ag.inner, ag.cons, u)
    # Rescale
    ag.cons .*= ag.scaler.scale_cons
    # Update (shifted) infeasibility error
    g♭ = ag.scaler.g_min
    g♯ = ag.scaler.g_max

    ag.λc = max.(0, ag.λ .+ ag.ρ .* (ag.cons .- g♯)) .+
            min.(0, ag.λ .+ ag.ρ .* (ag.cons .- g♭))
    ag.infeasibility .= max.(0, ag.cons .- g♯) .+ min.(0, ag.cons .- g♭)
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
    cx = ag.infeasibility
    # TODO: add multiplier
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
    model = base_nlp.model
    λ = base_nlp.λ
    # Import buffer (has been updated previously in update!)
    buffer = base_nlp.buffer
    # Import AutoDiff objects
    ∇gᵤ = jacobian(model, base_nlp.autodiff.Jgᵤ, buffer)
    ∂obj = base_nlp.autodiff.∇f
    jvx = ∂obj.jvₓ ; fill!(jvx, 0)
    jvu = ∂obj.jvᵤ ; fill!(jvu, 0)

    # compute gradient of objective
    ∂cost(model, ∂obj, buffer)
    jvx .+= scaler.scale_obj .* ∂obj.∇fₓ
    jvu .+= scaler.scale_obj .* ∂obj.∇fᵤ

    # compute gradient of penalties
    fr_ = 0
    for cons in base_nlp.constraints
        n = size_constraint(base_nlp.model, cons)
        mask = fr_+1:fr_+n
        cx = @view ag.λc[mask]
        v = cx .* scaler.scale_cons[mask]
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

function constraint!(ag::AugLagEvaluator, cons, u)
    @assert length(cons) == 0
    return
end

function jacobian!(ag::AugLagEvaluator, jac, u)
    @assert length(jac) == 0
    return
end

function jacobian_structure!(ag::AugLagEvaluator, rows, cols)
    @assert length(rows) == length(cols) == 0
end

primal_infeasibility!(ag::AugLagEvaluator, cons, u) = primal_infeasibility!(ag.inner, cons, u)
primal_infeasibility(ag::AugLagEvaluator, u) = primal_infeasibility(ag.inner, u)

function reset!(ag::AugLagEvaluator)
    reset!(ag.inner)
    empty!(ag.counter)
    fill!(ag.cons, 0)
    fill!(ag.infeasibility, 0)
    fill!(ag.λ, 0)
    fill!(ag.λc, 0)
end

