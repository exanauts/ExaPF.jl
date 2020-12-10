
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

    g_min, g_max = bounds(nlp, Constraints())
    cons = similar(g_min)
    cx = similar(g_min)
    λc = similar(g_min) ; fill!(λc, 0)
    λ = similar(g_min) ; fill!(λ, 0)

    scaler = scale ?  MaxScaler(nlp, u0) : MaxScaler(g_min, g_max)
    return AugLagEvaluator(nlp, cons, cx, c₀, λ, λc, scaler, NLPCounter())
end

n_variables(ag::AugLagEvaluator) = n_variables(ag.inner)
n_constraints(ag::AugLagEvaluator) = 0

# Getters
get(ag::AugLagEvaluator, attr::AbstractNLPAttribute) = get(ag.inner, attr)

# Initial position
initial(ag::AugLagEvaluator) = initial(ag.inner)

# Bounds
bounds(ag::AugLagEvaluator, ::Variables) = bounds(ag.inner, Variables())
bounds(ag::AugLagEvaluator, ::Constraints) = Float64[], Float64[]

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
    # Import buffer (has been updated previously in update!)
    buffer = get(base_nlp, PhysicalState())
    # Import AutoDiff objects
    autodiff = get(base_nlp, AutoDiffBackend())

    # Evaluate Jacobian of power flow equation on current u
    update_jacobian!(base_nlp, Control(), buffer)

    ∂obj = autodiff.∇f
    jvx = ∂obj.jvₓ ; fill!(jvx, 0)
    jvu = ∂obj.jvᵤ ; fill!(jvu, 0)

    # compute gradient of objective
    ∂cost(model, ∂obj, buffer)
    jvx .+= scaler.scale_obj .* ∂obj.∇fₓ
    jvu .+= scaler.scale_obj .* ∂obj.∇fᵤ

    # compute gradient of penalties
    constraints = get(base_nlp, Constraints())
    fr_ = 0
    for cons in constraints
        n = size_constraint(model, cons)
        mask = fr_+1:fr_+n
        cx = @view ag.λc[mask]
        v = cx .* scaler.scale_cons[mask]
        jtprod(model, cons, ∂obj, buffer, v)
        jvx .+= ∂obj.∇fₓ
        jvu .+= ∂obj.∇fᵤ
        fr_ += n
    end
    # Evaluate reduced gradient
    reduced_gradient!(base_nlp, grad, jvx, jvu)
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

