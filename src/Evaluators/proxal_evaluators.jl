
"""
    ProxALEvaluator{T} <: AbstractNLPEvaluator

Evaluator wrapping a `ReducedSpaceEvaluator` for use inside
dual decomposition algorithm.

"""
mutable struct ProxALEvaluator{T} <: AbstractNLPEvaluator
    inner::ReducedSpaceEvaluator{T}
    s_min::AbstractVector{T}
    s_max::AbstractVector{T}
    nu::Int
    ng::Int
    # Augmented penalties parameters
    time::Int
    τ::T
    λf::AbstractVector{T}
    λt::AbstractVector{T}
    ρf::T
    ρt::T
    pg_f::AbstractVector{T}
    pg_ref::AbstractVector{T}
    pg_t::AbstractVector{T}
    # Buffer
    ramp_link_prev::AbstractVector{T}
    ramp_link_next::AbstractVector{T}
end

# TODO: add constructor from PowerNetwork

n_variables(nlp::ProxALEvaluator) = n_variables(nlp.inner) + length(nlp.s_min)
n_constraints(nlp::ProxALEvaluator) = n_constraints(nlp.inner)

# Getters
get(nlp::ProxALEvaluator, attr::AbstractNLPAttribute) = get(nlp.inner, attr)

# Initial position
function initial(nlp::ProxALEvaluator)
    u0 = initial(nlp.model, Control())
    s0 = copy(nlp.s_min)
    return [u0; s0]
end

# Bounds
function bounds(nlp::ReducedSpaceEvaluator, ::Variables)
    u♭, u♯ = bounds(nlp.inner, Variables())
    return [u♭; nlp.s_min], [u♯; nlp.s_max]
end
bounds(nlp::ReducedSpaceEvaluator, ::Constraints) = bounds(nlp.inner, Constraints())

function update!(nlp::ProxALEvaluator, w)
    @assert length(w) == nlp.nu + nlp.ng
    u = w[1:nlp.nu]
    s = w[nlp.nu+1:end]
    conv = update!(nlp.inner, u)
    pg = get(nlp.inner, PS.ActivePower())
    # Update terms for augmented penalties
    nlp.ramp_link_prev .= nlp.pg_t .- pg .+ s
    nlp.ramp_link_next .= pg .- nlp.pg_t
    return conv
end

## Update penalty terms
function update_multipliers!(nlp::ProxALEvaluator, λf, λt)
    @assert length(λf) == length(λt) == nlp.ng
    copyto!(nlp.λf, λf)
    copyto!(nlp.λt, λt)
end

function update_primal!(nlp::ProxALEvaluator, pgf, pgc, pgt)
    @assert length(pgf) == length(pft) == length(pgc) == nlp.ng
    copyto!(nlp.pg_f, pfg)
    copyto!(nlp.pg_ref, pfc)
    copyto!(nlp.pg_t, pft)
end

## Objective
function objective(nlp::ProxALEvaluator, w)
    @assert length(w) == nlp.nu + nlp.ng
    u = w[1:nlp.nu]
    s = w[nlp.nu+1:end]
    pg = get(nlp, PS.ActivePower())
    # Operational costs
    cost = cost_production(nlp.inner.model, pg)
    # Augmented Lagrangian penalty
    cost += 0.5 * nlp.τ * xnorm(pg .- nlp.pg_ref)^2
    cost += dot(nlp.λf, nlp.ramp_link_prev)
    cost += dot(nlp.λt, ramp_link_next)
    cost += 0.5 * nlp.ρf * xnorm(ramp_link_prev)^2
    cost += 0.5 * nlp.ρt * xnorm(ramp_link_next)^2

    return cost
end

## Gradient
function gradient!(nlp::ProxALEvaluator, g, w)
    @assert length(w) == nlp.nu + nlp.ng
    u = w[1:nlp.nu]
    s = w[nlp.nu+1:end]
    pg = get(nlp.inner, PS.ActivePower())
    # Import buffer (has been updated previously in update!)
    buffer = get(nlp.inner, PhysicalState())
    # Import AutoDiff objects
    autodiff = get(nlp.inner, AutoDiffBackend())
    ∂obj = autodiff.∇f
    # Import model
    model = nlp.inner.model

    # Reduced gradient wrt u
    g_u = @view g[1:nlp.nu]
    ## Objective's coefficients
    coefs = model.costs_coefficients
    c3 = @view coefs[:, 3]
    c4 = @view coefs[:, 4]
    ## Seed left-hand side vector
    ∂obj.∂pg .= c3 .+ 2.0 .* c4 .* pg
    ∂obj.∂pg .+= (nlp.λt .- nlp.λf)
    ∂obj.∂pg .-= nlp.ρf .* nlp.ramp_link_prev
    ∂obj.∂pg .+= nlp.ρt .* nlp.ramp_link_next
    ∂obj.∂pg .+= nlp.τ .* (pg .- nlp.pg_ref)

    ## Evaluate conjointly
    # ∇fₓ = v' J,  with J = ∂pg / ∂x
    # ∇fᵤ = v' J,  with J = ∂pg / ∂u
    put(model, PS.Generator(), PS.ActivePower(), ∂obj, buffer)

    ## Evaluation of reduced gradient
    reduced_gradient!(nlp.inner, g_u, ∂obj.∇fₓ, ∂obj.∇fᵤ)

    # Gradient wrt s
    g_s = @view g[nlp.nu+1:end]
    g_s .+= nlp.λf .+ nlp.ρf .* (nlp.pg_t .- pg .+ s)
    return nothing
end

function constraint!(nlp::ProxALEvaluator, cons, w)
    @assert length(w) == nlp.nu + nlp.ng
    u = w[1:nlp.nu]
    constraint!(nlp.inner, cons, u)
end

## Jacobian
function jacobian_structure!(nlp::ProxALEvaluator, rows, cols)
    jacobian_structure!(nlp.inner, rows, cols)
end

function jacobian!(nlp::ProxALEvaluator, jac, w)
    @assert length(w) == nlp.nu + nlp.ng
    u = w[1:nlp.nu]
    jacobian!(nlp, jac, u)
end

## Transpose Jacobian-vector product
function jtprod!(nlp::ProxALEvaluator, cons, jv, w, v; start=1)
    @assert length(w) == nlp.nu + nlp.ng
    u = w[1:nlp.nu]
    jtprod!(nlp.inner, cons, jv, u, v; start=start)
end
function jtprod!(nlp::ProxALEvaluator, jv, w, v)
    @assert length(w) == nlp.nu + nlp.ng
    u = w[1:nlp.nu]
    jtprod!(nlp.inner, jv, u, v)
end


## Utils function
function primal_infeasibility!(nlp::ProxALEvaluator, cons, w)
    @assert length(w) == nlp.nu + nlp.ng
    u = w[1:nlp.nu]
    return primal_infeasibility(nlp.inner, cons, u)
end
function primal_infeasibility(nlp::ProxALEvaluator, w)
    @assert length(w) == nlp.nu + nlp.ng
    u = w[1:nlp.nu]
    return primal_infeasibility!(nlp.inner, cons, u)
end

function reset!(nlp::ProxALEvaluator)
    reset!(nlp.inner)
end
