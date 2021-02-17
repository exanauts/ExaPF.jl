
@enum(ProxALTime,
    Origin,
    Final,
    Normal,
)

abstract type AbstractTimeStep end
struct Current <: AbstractTimeStep end
struct Previous <: AbstractTimeStep end
struct Next <: AbstractTimeStep end


"""
    ProxALEvaluator{T, VI, VT, MT} <: AbstractNLPEvaluator

Evaluator wrapping a `ReducedSpaceEvaluator` for use inside the
decomposition algorithm implemented in [ProxAL.jl](https://github.com/exanauts/ProxAL.jl).

"""
mutable struct ProxALEvaluator{T, VI, VT, MT} <: AbstractNLPEvaluator
    inner::ReducedSpaceEvaluator{T, VI, VT, MT}
    s_min::VT
    s_max::VT
    nu::Int
    ng::Int
    # Augmented penalties parameters
    time::ProxALTime
    scale_objective::T
    τ::T
    λf::VT
    λt::VT
    ρf::T
    ρt::T
    pg_f::VT
    pg_ref::VT
    pg_t::VT
    # Buffer
    ramp_link_prev::VT
    ramp_link_next::VT
end
function ProxALEvaluator(
    nlp::ReducedSpaceEvaluator{T, VI, VT, MT},
    time::ProxALTime;
    τ=0.1, ρf=0.1, ρt=0.1, scale_obj=1.0,
) where {T, VI, VT, MT}
    nu = n_variables(nlp)
    ng = get(nlp, PS.NumberOfGenerators())

    s_min = xzeros(VT, ng)
    s_max = xones(VT, ng)
    λf = xzeros(VT, ng)
    λt = xzeros(VT, ng)

    pgf = xzeros(VT, ng)
    pgc = xzeros(VT, ng)
    pgt = xzeros(VT, ng)

    ramp_link_prev = xzeros(VT, ng)
    ramp_link_next = xzeros(VT, ng)

    return ProxALEvaluator(
        nlp, s_min, s_max, nu, ng, time, scale_obj, τ, λf, λt, ρf, ρt,
        pgf, pgc, pgt, ramp_link_prev, ramp_link_next,
    )
end
function ProxALEvaluator(
    pf::PS.PowerNetwork,
    time::ProxALTime;
    device=KA.CPU(),
    options...
)
    # Build network polar formulation
    model = PolarForm(pf, device)
    # Build reduced space evaluator
    x = initial(model, State())
    u = initial(model, Control())
    nlp = ReducedSpaceEvaluator(model, x, u; options...)
    return ProxALEvaluator(nlp, time)
end
function ProxALEvaluator(
    datafile::String;
    time::ProxALTime=Normal,
    device=KA.CPU(),
    options...
)
    nlp = ReducedSpaceEvaluator(datafile; device=device, options...)
    return ProxALEvaluator(nlp, time; options...)
end

n_variables(nlp::ProxALEvaluator) = nlp.nu + nlp.ng
n_constraints(nlp::ProxALEvaluator) = n_constraints(nlp.inner)

constraints_type(::ProxALEvaluator) = :inequality

# Getters
get(nlp::ProxALEvaluator, attr::AbstractNLPAttribute) = get(nlp.inner, attr)
get(nlp::ProxALEvaluator, attr::AbstractVariable) = get(nlp.inner, attr)
get(nlp::ProxALEvaluator, attr::PS.AbstractNetworkValues) = get(nlp.inner, attr)
get(nlp::ProxALEvaluator, attr::PS.AbstractNetworkAttribute) = get(nlp.inner, attr)

# Setters
function setvalues!(nlp::ProxALEvaluator, attr::PS.AbstractNetworkValues, values)
    setvalues!(nlp.inner, attr, values)
end
function transfer!(nlp::ProxALEvaluator, vm, va, pg, qg)
    transfer!(nlp.inner, vm, va, pg, qg)
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
    u = @view w[1:nlp.nu]
    s = @view w[nlp.nu+1:end]
    conv = update!(nlp.inner, u)
    pg = get(nlp.inner, PS.ActivePower())
    # Update terms for augmented penalties
    if nlp.time != Origin
        nlp.ramp_link_prev .= nlp.pg_f .- pg .+ s
    end
    if nlp.time != Final
        nlp.ramp_link_next .= pg .- nlp.pg_t
    end
    return conv
end

## Update penalty terms
function update_multipliers!(nlp::ProxALEvaluator, ::Current, λt)
    copyto!(nlp.λf, λt)
end
function update_multipliers!(nlp::ProxALEvaluator, ::Next, λt)
    copyto!(nlp.λt, λt)
end

function update_primal!(nlp::ProxALEvaluator, ::Previous, pgk)
    copyto!(nlp.pg_f, pgk)
end
function update_primal!(nlp::ProxALEvaluator, ::Current, pgk)
    copyto!(nlp.pg_ref, pgk)
end
function update_primal!(nlp::ProxALEvaluator, ::Next, pgk)
    copyto!(nlp.pg_t, pgk)
end

## Objective
function objective(nlp::ProxALEvaluator, w)
    u = @view w[1:nlp.nu]
    s = @view w[nlp.nu+1:end]
    pg = get(nlp.inner, PS.ActivePower())
    # Operational costs
    cost = nlp.scale_objective * cost_production(nlp.inner.model, pg)
    # Augmented Lagrangian penalty
    cost += 0.5 * nlp.τ * xnorm(pg .- nlp.pg_ref)^2
    if nlp.time != Origin
        cost += dot(nlp.λf, nlp.ramp_link_prev)
        cost += 0.5 * nlp.ρf * xnorm(nlp.ramp_link_prev)^2
    end
    if nlp.time != Final
        cost += dot(nlp.λt, nlp.ramp_link_next)
        cost += 0.5 * nlp.ρt * xnorm(nlp.ramp_link_next)^2
    end
    return cost
end

## Gradient
update_jacobian!(nlp::ProxALEvaluator, ::Control) = update_jacobian!(nlp.inner, Control())

function full_gradient!(nlp::ProxALEvaluator, jvx, jvu, w)
    # Import model
    model = nlp.inner.model
    # Import buffer (has been updated previously in update!)
    buffer = get(nlp.inner, PhysicalState())
    # Import AutoDiff objects
    autodiff = get(nlp.inner, AutoDiffBackend())
    ∂obj = autodiff.∇f
    # Scaling
    scale_obj = nlp.scale_objective
    # Current active power
    pg = get(nlp.inner, PS.ActivePower())

    u = @view w[1:nlp.nu]

    ## Objective's coefficients
    coefs = model.costs_coefficients
    c3 = @view coefs[:, 3]
    c4 = @view coefs[:, 4]
    ## Seed left-hand side vector
    ∂obj.∂pg .= scale_obj .* (c3 .+ 2.0 .* c4 .* pg)
    if nlp.time != Origin
        ∂obj.∂pg .-= nlp.λf
        ∂obj.∂pg .-= nlp.ρf .* nlp.ramp_link_prev
    end
    if nlp.time != Final
        ∂obj.∂pg .+= nlp.λt
        ∂obj.∂pg .+= nlp.ρt .* nlp.ramp_link_next
    end
    ∂obj.∂pg .+= nlp.τ .* (pg .- nlp.pg_ref)

    ## Evaluate conjointly
    # ∇fₓ = v' * J,  with J = ∂pg / ∂x
    # ∇fᵤ = v' * J,  with J = ∂pg / ∂u
    put(model, PS.Generators(), PS.ActivePower(), ∂obj, buffer)

    copyto!(jvx, ∂obj.∇fₓ)
    copyto!(jvu, ∂obj.∇fᵤ)
end

function gradient_slack!(nlp::ProxALEvaluator, grad, w)
    # Gradient wrt s
    g_s = @view grad[nlp.nu+1:end]
    if nlp.time != Origin
        g_s .= nlp.λf .+ nlp.ρf .* nlp.ramp_link_prev
    else
        g_s .= 0.0
    end
end

function reduced_gradient!(nlp::ProxALEvaluator, grad, jvx, jvu, w)
    g = @view grad[1:nlp.nu]
    reduced_gradient!(nlp.inner, g, jvx, jvu, w)
    gradient_slack!(nlp, grad, w)
end

## Gradient
function gradient!(nlp::ProxALEvaluator, g, w)
    # Import AutoDiff objects
    autodiff = get(nlp.inner, AutoDiffBackend())
    ∂obj = autodiff.∇f

    jvu = ∂obj.∇fᵤ ; jvx = ∂obj.∇fₓ
    fill!(jvx, 0)  ; fill!(jvu, 0)
    full_gradient!(nlp, jvx, jvu, w)

    # Start to update Control Jacobian in reduced model
    update_jacobian!(nlp, Control())
    ## Evaluation of reduced gradient
    reduced_gradient!(nlp, g, jvx, jvu, w)
    return nothing
end

function constraint!(nlp::ProxALEvaluator, cons, w)
    u = @view w[1:nlp.nu]
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
## ProxAL does not add any constraint to the reduced model
function jtprod!(nlp::ProxALEvaluator, jv, w, v)
    u = @view w[1:nlp.nu]
    jvu = @view jv[1:nlp.nu]
    jtprod!(nlp.inner, jvu, u, v)
end
function full_jtprod!(nlp::ProxALEvaluator, jvx, jvu, w, v)
    u = @view w[1:nlp.nu]
    full_jtprod!(nlp.inner, jvx, jvu, u, v)
end

function ojtprod!(nlp::ProxALEvaluator, jv, u, σ, v)
    autodiff = get(nlp, AutoDiffBackend())
    ∂obj = autodiff.∇f
    jvx = ∂obj.jvₓ ; fill!(jvx, 0)
    jvu = ∂obj.jvᵤ ; fill!(jvu, 0)
    # compute gradient of objective
    full_gradient!(nlp, jvx, jvu, u)
    jvx .*= σ
    jvu .*= σ
    # compute transpose Jacobian vector product of constraints
    full_jtprod!(nlp, jvx, jvu, u, v)
    # Evaluate gradient in reduced space
    update_jacobian!(nlp, Control())
    reduced_gradient!(nlp, jv, jvx, jvu, u)
end

## Utils function
function reset!(nlp::ProxALEvaluator)
    reset!(nlp.inner)
    # Reset multipliers
    fill!(nlp.λf, 0)
    fill!(nlp.λt, 0)
    # Reset proximal centers
    fill!(nlp.pg_f, 0)
    fill!(nlp.pg_ref, 0)
    fill!(nlp.pg_t, 0)
    # Reset buffers
    fill!(nlp.ramp_link_prev, 0)
    fill!(nlp.ramp_link_next, 0)
end

function primal_infeasibility!(nlp::ProxALEvaluator, cons, w)
    @assert length(w) == nlp.nu + nlp.ng
    u = @view w[1:nlp.nu]
    return primal_infeasibility!(nlp.inner, cons, u)
end
function primal_infeasibility(nlp::ProxALEvaluator, w)
    @assert length(w) == nlp.nu + nlp.ng
    u = @view w[1:nlp.nu]
    return primal_infeasibility(nlp.inner, u)
end
