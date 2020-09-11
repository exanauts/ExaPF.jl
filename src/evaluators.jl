
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

    constraints::Array{Function, 1}
    g_min::AbstractVector{T}
    g_max::AbstractVector{T}

    ad::ADFactory
    precond::Precondition.AbstractPreconditioner
    solver::String
    ε_tol::Float64
end

function ReducedSpaceEvaluator(model, x, u, p;
                               constraints=Function[state_constraint],
                               ε_tol=1e-12, solver="default", npartitions=2,
                               verbose_level=VERBOSE_LEVEL_NONE)
    # Build up AD factory
    jx, ju = init_ad_factory(model, x, u, p)
    ad = ADFactory(jx, ju)
    # Init preconditioner if needed for iterative linear algebra
    precond = Iterative.init_preconditioner(jx.J, solver, npartitions, model.device)

    u_min, u_max = bounds(model, Control())
    x_min, x_max = bounds(model, State())

    MT = model.AT
    g_min = MT{eltype(x), 1}()
    g_max = MT{eltype(x), 1}()
    for cons in constraints
        cb, cu = bounds(model, cons)
        append!(g_min, cb)
        append!(g_max, cu)
    end

    return ReducedSpaceEvaluator(model, x, p, x_min, x_max, u_min, u_max,
                                 constraints, g_min, g_max,
                                 ad, precond, solver, ε_tol)
end

n_variables(nlp::ReducedSpaceEvaluator) = length(nlp.u_min)
n_constraints(nlp::ReducedSpaceEvaluator) = length(nlp.g_min)

function update!(nlp::ReducedSpaceEvaluator, u; verbose_level=0)
    x₀ = nlp.x
    jac_x = nlp.ad.Jgₓ
    # Get corresponding point on the manifold
    xk, conv = powerflow(nlp.model, jac_x, x₀, u, nlp.p, tol=nlp.ε_tol;
                         solver=nlp.solver, preconditioner=nlp.precond, verbose_level=verbose_level)
    copy!(nlp.x, xk)
    return conv
end

function objective(nlp::ReducedSpaceEvaluator, u)
    cost = cost_production(nlp.model, nlp.x, u, nlp.p)
    # TODO: determine if we should include λ' * g(x, u), even if ≈ 0
    return cost
end

# Private function to compute adjoint (should be inlined)
_adjoint(J, y) = - J' \ y
function _adjoint(J::CuSparseMatrixCSR{T}, y::CuVector{T}) where T
    # TODO: we SHOULD find a most efficient implementation
    Jt = CuArray(J') |> sparse
    λk = similar(y)
    return CUSOLVER.csrlsvqr!(Jt, -y, λk, 1e-8, one(Cint), 'O')
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
    λₖ = _adjoint(∇gₓ, ∇fₓ)
    # compute reduced gradient
    g .= ∇fᵤ + (∇gᵤ')*λₖ
    return nothing
end

function constraint!(nlp::ReducedSpaceEvaluator, g, u)
    xₖ = nlp.x
    # First: state constraint
    mf = 1
    mt = 0
    for cons in nlp.constraints
        m_ = size_constraint(nlp.model, cons)
        mt += m_
        cons_ = @view(g[mf:mt])
        cons(nlp.model, cons_, xₖ, u, nlp.p)
        mf += m_
    end
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

function jacobian!(nlp::ReducedSpaceEvaluator, u)
    xₖ = nlp.x
    ∇gₓ = nlp.ad.Jgₓ.J
    ∇gᵤ = nlp.ad.Jgᵤ.J
    nₓ = length(xₖ)
    m = n_constraints(nlp)
    n = length(u)
    MT = nlp.model.AT
    J = MT{eltype(u), 2}(undef, m, n)
    cnt = 1
    for cons in nlp.constraints
        mc_ = size_constraint(nlp.model, cons)
        g = MT{eltype(u), 1}(undef, mc_)
        fill!(g, 0)
        cons_x(g, x_) = cons(nlp.model, g, x_, u, nlp.p; V=eltype(x_))
        cons_u(g, u_) = cons(nlp.model, g, xₖ, u_, nlp.p; V=eltype(u_))
        Jₓ = ForwardDiff.jacobian(cons_x, g, xₖ)
        Jᵤ = ForwardDiff.jacobian(cons_u, g, u)
        for ix in 1:mc_
            rhs = Jₓ[ix, :]
            λ = _adjoint(∇gₓ, rhs)
            J[cnt, :] .= Jᵤ[ix, :] + ∇gᵤ' * λ
            cnt += 1
        end
    end
    return J
end
