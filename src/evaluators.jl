
import Base: show

# WIP: definition of AD factory
abstract type AbstractADFactory end

struct ADFactory <: AbstractADFactory
    Jgₓ::AD.StateJacobianAD
    Jgᵤ::AD.DesignJacobianAD
end

abstract type AbstractNLPEvaluator end

struct ReducedSpaceEvaluator{T} <: AbstractNLPEvaluator
    model::AbstractFormulation
    x::AbstractVector{T}
    p::AbstractVector{T}

    x_min::AbstractVector{T}
    x_max::AbstractVector{T}
    u_min::AbstractVector{T}
    u_max::AbstractVector{T}

    ad::ADFactory
    ε_tol::Float64
end

function ReducedSpaceEvaluator(model, x, u, p; ε_tol=1e-12)
    jx, ju = init_ad_factory(model, x, u, p)
    ad = ADFactory(jx, ju)
    u_min, u_max = bounds(model, Control())
    x_min, x_max = bounds(model, State())
    return ReducedSpaceEvaluator(model, x, p, x_min, x_max, u_min, u_max,
                                 ad, ε_tol)
end

n_variables(nlp::ReducedSpaceEvaluator) = length(nlp.u_min)
# n_constraints(nlp::ReducedSpaceEvaluator) = length(nlp.u)

function update!(nlp::ReducedSpaceEvaluator, u)
    x₀ = nlp.x
    jac_x = nlp.ad.Jgₓ
    # Get corresponding point on the manifold
    xk, conv = powerflow(nlp.model, jac_x, x₀, u, nlp.p, tol=nlp.ε_tol)
    copy!(nlp.x, xk)
end

function objective(nlp::ReducedSpaceEvaluator, u)
    cost = cost_production(nlp.model, nlp.x, u, nlp.p)
    # TODO: determine if we should include λ' * g(x, u), even if ≈ 0
    return cost
end

function gradient!(nlp::ReducedSpaceEvaluator, g, u)
    xₖ = nlp.x
    # TODO: could we move this in the AD factory?
    cost_x = x_ -> cost_production(nlp.model, x_, u, nlp.p; V=eltype(x_))
    cost_u = u_ -> cost_production(nlp.model, xₖ, u_, nlp.p; V=eltype(u_))
    fdCdx = x_ -> ForwardDiff.gradient(cost_x, x_)
    fdCdu = u_ -> ForwardDiff.gradient(cost_u, u_)
    ∇gₓ = nlp.ad.Jgₓ.J
    # Evaluate Jacobian of power flow equation on current u
    ∇gᵤ = jacobian(nlp.model, nlp.ad.Jgᵤ, xₖ, u, nlp.p)
    ∇fₓ = fdCdx(xₖ)
    ∇fᵤ = fdCdu(u)
    # Update adjoint
    λₖ = -(∇gₓ'\∇fₓ)
    # compute reduced gradient
    g .= ∇fᵤ + (∇gᵤ')*λₖ
    return nothing
end

function constraint!(nlp::ReducedSpaceEvaluator, cons, u)
end

#TODO: return sparsity pattern there, currently return dense pattern
function jacobian_structure!(nlp::ReducedSpaceEvaluator, rows, cols)
    m, n = n_constraints(nlp), n_variables(nlp)
    idx = 1
    for c in 1:m #number of constraints
        for i in 1:n # number of variables
            rows[idx] = c ; cols[idx] = i
            idx += 1
        end
    end
end

function jacobian!(nlp::ReducedSpaceEvaluator, jac, u)
    nx = length(xk)
    n = length(u)
    J = zeros(m, n)
    rhs = zeros(nx)
    λ = zeros(nx)
    # Evaluate reduced Jacobian for bounds on vmag_{pq}
    for ix in 1:npq
        rhs .= 0.0
        rhs[ix] = 1.0
        λ .= - ∇gₓ' \ rhs
        J[ix, :] .= ∇gᵤ' * λ
    end
    # Evaluate reduced Jacobian for bounds on p_{ref}
    gg_x(x_) = gₚ(model, x_, u, p; V=eltype(x_))
    gg_u(u_) = gₚ(model, xk, u_, p; V=eltype(u_))
    jac_p_x = ForwardDiff.jacobian(gg_x, xk)
    jac_p_u = ForwardDiff.jacobian(gg_u, u)
    for ix in 1:(nref + npv)
        rhs = jac_p_x[ix, :]
        λ .= - ∇gₓ' \ rhs
        J[npq + ix, :] .= jac_p_u[ix, :] + ∇gᵤ' * λ
    end
    # Evaluate reduced Jacobian for flow limits
    hh_x(x_) = flow_limit(model, x_, u, p; T=eltype(x_))
    hh_u(u_) = flow_limit(model, xk, u_, p; T=eltype(u_))
    jac_h_x = ForwardDiff.jacobian(hh_x, xk)
    jac_h_u = ForwardDiff.jacobian(hh_u, u)
    _shift = npq + nref + npv
    for ix in 1:2*nlines
        rhs = jac_h_x[ix, :]
        λ .= - ∇gₓ' \ rhs
        J[_shift + ix, :] .= jac_h_u[ix, :] + ∇gᵤ' * λ
    end
end
