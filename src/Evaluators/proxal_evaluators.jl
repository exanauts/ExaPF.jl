
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
mutable struct ProxALEvaluator{T, VI, VT, MT, Pullback, Hess} <: AbstractNLPEvaluator
    inner::ReducedSpaceEvaluator{T, VI, VT, MT}
    obj_stack::Pullback
    hessian_obj::Hess
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
end
function ProxALEvaluator(
    nlp::ReducedSpaceEvaluator{T, VI, VT, MT},
    time::ProxALTime;
    τ=0.1, ρf=0.1, ρt=0.1, scale_obj=1.0, want_hessian=true,
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

    intermediate = (
        s = similar(s_min),
        t = Int(time),
        σ = scale_obj,
        τ = τ,
        λf = λf,
        λt = λt,
        ρf = ρf,
        ρt = ρt,
        p1 = pgf,
        p2 = pgc,
        p3 = pgt,
    )

    pbm = pullback_ramping(nlp.model, intermediate)

    hess = nothing
    if want_hessian
        hess = AutoDiff.Hessian(nlp.model, cost_penalty_ramping_constraints; tape=pbm)
    end
    return ProxALEvaluator(
        nlp, pbm, hess, s_min, s_max, nu, ng, time, scale_obj,
        τ, λf, λt, ρf, ρt,
        pgf, pgc, pgt,
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
    nlp = ReducedSpaceEvaluator(model; options...)
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
has_hessian(::ProxALEvaluator) = true
backend(nlp::ProxALEvaluator) = backend(nlp.inner)

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
    copyto!(nlp.obj_stack.intermediate.s, s)
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
    model = nlp.inner.model
    buffer = get(nlp.inner, PhysicalState())
    return cost_penalty_ramping_constraints(
        model, buffer, s, Int(nlp.time),
        nlp.scale_objective, nlp.τ, nlp.λf, nlp.λt, nlp.ρf, nlp.ρt, nlp.pg_f, nlp.pg_ref, nlp.pg_t
    )
end

## Gradient
function full_gradient!(nlp::ProxALEvaluator, jx, ju, w)
    buffer = get(nlp.inner, PhysicalState())
    ∂obj = nlp.obj_stack
    # Evaluate adjoint of cost function and update inplace AdjointStackObjective
    gradient_objective!(nlp.inner.model, ∂obj, buffer)
    copyto!(jx, ∂obj.stack.∇fₓ)
    copyto!(ju, ∂obj.stack.∇fᵤ)
end

function gradient_slack!(nlp::ProxALEvaluator, grad, w)
    s = @view w[nlp.nu+1:end]
    pg = get(nlp.inner, PS.ActivePower())
    # Gradient wrt s
    g_s = @view grad[nlp.nu+1:end]
    if nlp.time != Origin
        g_s .= nlp.λf .+ nlp.ρf .* (nlp.pg_f .- pg .+ s)
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
    ∂obj = nlp.obj_stack
    jvu = ∂obj.stack.∇fᵤ ; jvx = ∂obj.stack.∇fₓ
    fill!(jvx, 0)  ; fill!(jvu, 0)
    full_gradient!(nlp, jvx, jvu, w)

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
    m = n_constraints(nlp)
    nnj = length(jac)
    u = @view w[1:nlp.nu]
    J = reshape(jac, m, div(nnj, m))
    Jᵤ = @view J[:, 1:nlp.nu]
    jacobian!(nlp.inner, Jᵤ, u)
end

function jprod!(nlp::ProxALEvaluator, jv, w, v)
    u = @view w[1:nlp.nu]
    vu = @view v[1:nlp.nu]
    jprod!(nlp.inner, jv, u, vu)
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
    ∂obj = nlp.obj_stack
    jvx = ∂obj.stack.jvₓ ; fill!(jvx, 0)
    jvu = ∂obj.stack.jvᵤ ; fill!(jvu, 0)
    # compute gradient of objective
    full_gradient!(nlp, jvx, jvu, u)
    jvx .*= σ
    jvu .*= σ
    # compute transpose Jacobian vector product of constraints
    full_jtprod!(nlp, jvx, jvu, u, v)
    # Evaluate gradient in reduced space
    reduced_gradient!(nlp, jv, jvx, jvu, u)
end

#=
    For ProxAL, we have:
    H = [ H_xx  H_ux  J_x' ]
        [ H_xu  H_uu  J_u' ]
        [ J_x   J_u   ρ I  ]

    so, if `v = [v_x; v_u; v_s]`, we get

    H * v = [ H_xx v_x  +   H_ux v_u  +  J_x' v_s ]
            [ H_xu v_x  +   H_uu v_u  +  J_u' v_s ]
            [  J_x v_x  +    J_u v_u  +   ρ I     ]

=#
function hessprod!(nlp::ProxALEvaluator, hessvec, w, v)
    @assert nlp.inner.has_hessian
    @assert nlp.inner.has_jacobian

    model = nlp.inner.model
    nx = get(model, NumberOfState())
    nu = get(model, NumberOfControl())

    u = @view w[1:nlp.nu]
    vᵤ = @view v[1:nlp.nu]
    vₛ = @view v[1+nlp.nu:end]

    fill!(hessvec, 0.0)

    hvu = @view hessvec[1:nlp.nu]
    hvs = @view hessvec[1+nlp.nu:end]

    ## OBJECTIVE HESSIAN
    σ = 1.0
    hessprod!(nlp.inner, hvu, u, vᵤ)

    # Contribution of slack node
    if nlp.time != Origin
        hvs .+= nlp.ρf .* vₛ
        # TODO: implement block corresponding to Jacobian
        # and transpose-Jacobian
    end
    return
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
