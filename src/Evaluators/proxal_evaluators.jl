
@enum(ProxALTime,
    Origin,
    Final,
    Normal,
)

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
    time::ProxALTime
    scale_objective::T
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

function ProxALEvaluator(
    nlp::ReducedSpaceEvaluator,
    time::ProxALTime;
    τ=0.1, ρf=0.1, ρt=0.1, scale_obj=1.0,
)
    S = type_array(nlp)

    nu = n_variables(nlp)
    ng = get(nlp, PS.NumberOfGenerators())

    s_min = xzeros(S, ng)
    # TODO: fix s_max properly
    s_max = xones(S, ng)
    λf = xzeros(S, ng)
    λt = xzeros(S, ng)

    pgf = xzeros(S, ng)
    pgc = xzeros(S, ng)
    pgt = xzeros(S, ng)

    ramp_link_prev = xzeros(S, ng)
    ramp_link_next = xzeros(S, ng)

    return ProxALEvaluator(
        nlp, s_min, s_max, nu, ng, time, scale_obj, τ, λf, λt, ρf, ρt,
        pgf, pgc, pgt, ramp_link_prev, ramp_link_next,
    )
end

# TODO: add constructor from PowerNetwork
function ProxALEvaluator(
    pf::PS.PowerNetwork,
    time::ProxALTime;
    device=CPU(),
)
    # Build network polar formulation
    model = PolarForm(pf, device)
    # Build reduced space evaluator
    x = initial(model, State())
    p = initial(model, Parameters())
    u = initial(model, Control())
    nlp = ReducedSpaceEvaluator(model, x, u, p)
    return ProxALEvaluator(nlp, time)
end

n_variables(nlp::ProxALEvaluator) = nlp.nu + nlp.ng
n_constraints(nlp::ProxALEvaluator) = n_constraints(nlp.inner)

# Getters
get(nlp::ProxALEvaluator, attr::AbstractNLPAttribute) = get(nlp.inner, attr)

# Setters
function setvalues!(nlp::ProxALEvaluator, attr::PS.AbstractNetworkValues, values)
    setvalues!(nlp.inner, attr, values)
end

# Initial position
function initial(nlp::ProxALEvaluator)
    u0 = initial(nlp.inner)
    s0 = copy(nlp.s_min)
    return [u0; s0]
end

# Bounds
function bounds(nlp::ProxALEvaluator, ::Variables)
    u♭, u♯ = bounds(nlp.inner, Variables())
    return [u♭; nlp.s_min], [u♯; nlp.s_max]
end
bounds(nlp::ProxALEvaluator, ::Constraints) = bounds(nlp.inner, Constraints())

function update!(nlp::ProxALEvaluator, w)
    u = w[1:nlp.nu]
    s = w[nlp.nu+1:end]
    conv = update!(nlp.inner, u)
    pg = get(nlp.inner, PS.ActivePower())
    # Update terms for augmented penalties
    nlp.ramp_link_prev .= nlp.pg_f .- pg .+ s
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
    @assert length(pgf) == length(pgc) == length(pgt) == nlp.ng
    copyto!(nlp.pg_f, pgf)
    copyto!(nlp.pg_ref, pgc)
    copyto!(nlp.pg_t, pgt)
end

## Objective
function objective(nlp::ProxALEvaluator, w)
    u = w[1:nlp.nu]
    s = w[nlp.nu+1:end]
    pg = get(nlp.inner, PS.ActivePower())
    # Operational costs
    cost = nlp.scale_objective * cost_production(nlp.inner.model, pg)
    # Augmented Lagrangian penalty
    cost += 0.5 * nlp.τ * xnorm(pg .- nlp.pg_ref)^2
    cost += dot(nlp.λf, nlp.ramp_link_prev)
    cost += dot(nlp.λt, nlp.ramp_link_next)
    cost += 0.5 * nlp.ρf * xnorm(nlp.ramp_link_prev)^2
    cost += 0.5 * nlp.ρt * xnorm(nlp.ramp_link_next)^2

    return cost
end

## Gradient
function gradient!(nlp::ProxALEvaluator, g, w)
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
    # Scaling
    scale_obj = nlp.scale_objective

    # Start to update Control Jacobian in reduced model
    update_jacobian!(nlp.inner, Control())

    # Reduced gradient wrt u
    g_u = @view g[1:nlp.nu]
    ## Objective's coefficients
    coefs = model.costs_coefficients
    c3 = @view coefs[:, 3]
    c4 = @view coefs[:, 4]
    ## Seed left-hand side vector
    ∂obj.∂pg .= scale_obj .* (c3 .+ 2.0 .* c4 .* pg)
    ∂obj.∂pg .+= (nlp.λt .- nlp.λf)
    ∂obj.∂pg .-= nlp.ρf .* nlp.ramp_link_prev
    ∂obj.∂pg .+= nlp.ρt .* nlp.ramp_link_next
    ∂obj.∂pg .+= nlp.τ .* (pg .- nlp.pg_ref)

    ## Evaluate conjointly
    # ∇fₓ = v' * J,  with J = ∂pg / ∂x
    # ∇fᵤ = v' * J,  with J = ∂pg / ∂u
    fill!(∂obj.∇fᵤ, 0)
    fill!(∂obj.∇fₓ, 0)
    put(model, PS.Generator(), PS.ActivePower(), ∂obj, buffer)

    ## Evaluation of reduced gradient
    reduced_gradient!(nlp.inner, g_u, ∂obj.∇fₓ, ∂obj.∇fᵤ)

    # Gradient wrt s
    g_s = @view g[nlp.nu+1:end]
    g_s .= nlp.λf .+ nlp.ρf .* nlp.ramp_link_prev
    return nothing
end

function constraint!(nlp::ProxALEvaluator, cons, w)
    u = w[1:nlp.nu]
    constraint!(nlp.inner, cons, u)
end

function jacobian_structure(nlp::ProxALEvaluator)
    m, nu = n_constraints(nlp), nlp.nu
    nnzj = m * nu
    rows = zeros(Int, nnzj)
    cols = zeros(Int, nnzj)
    jacobian_structure!(nlp.inner, rows, cols)
    return rows, cols
end

## Jacobian
function jacobian_structure!(nlp::ProxALEvaluator, rows, cols)
    jacobian_structure!(nlp.inner, rows, cols)
end

function jacobian!(nlp::ProxALEvaluator, jac, w)
    u = @view w[1:nlp.nu]
    Jᵤ = @view jac[:, 1:nlp.nu]
    jacobian!(nlp.inner, Jᵤ, u)
end

## Transpose Jacobian-vector product
function jtprod!(nlp::ProxALEvaluator, cons, jv, w, v; start=1)
    u = w[1:nlp.nu]
    jvu = jv[1:nlp.nu]
    jtprod!(nlp.inner, cons, jvu, u, v; start=start)
end
function jtprod!(nlp::ProxALEvaluator, jv, w, v)
    u = w[1:nlp.nu]
    jvu = jv[1:nlp.nu]
    jtprod!(nlp.inner, jvu, u, v)
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

