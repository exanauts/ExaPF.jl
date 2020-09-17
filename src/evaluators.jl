
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
    λ::AbstractVector{T}

    x_min::AbstractVector{T}
    x_max::AbstractVector{T}
    u_min::AbstractVector{T}
    u_max::AbstractVector{T}

    constraints::Array{Function, 1}
    g_min::AbstractVector{T}
    g_max::AbstractVector{T}

    network_cache::NetworkState
    ad::ADFactory
    precond::Precondition.AbstractPreconditioner
    solver::String
    ε_tol::Float64
end

function ReducedSpaceEvaluator(model, x, u, p;
                               constraints=Function[state_constraint],
                               ε_tol=1e-12, solver="default", npartitions=2,
                               verbose_level=VERBOSE_LEVEL_NONE)
    # First, build up a network cache
    network_cache = NetworkState(model)
    # Initiate adjoint
    λ = similar(x)
    # Build up AD factory
    jx, ju = init_ad_factory(model, network_cache)
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

    return ReducedSpaceEvaluator(model, x, p, λ, x_min, x_max, u_min, u_max,
                                 constraints, g_min, g_max,
                                 network_cache,
                                 ad, precond, solver, ε_tol)
end

n_variables(nlp::ReducedSpaceEvaluator) = length(nlp.u_min)
n_constraints(nlp::ReducedSpaceEvaluator) = length(nlp.g_min)

function update!(nlp::ReducedSpaceEvaluator, u; verbose_level=0)
    x₀ = nlp.x
    jac_x = nlp.ad.Jgₓ
    # Transfer x, u, p into the network cache
    transfer!(nlp.model, nlp.network_cache, nlp.x, u, nlp.p)
    # Get corresponding point on the manifold
    conv = powerflow(nlp.model, jac_x, nlp.network_cache, tol=nlp.ε_tol;
                         solver=nlp.solver, preconditioner=nlp.precond, verbose_level=verbose_level)
    get!(nlp.model, State(), nlp.x, nlp.network_cache)
    refresh!(nlp.model, PS.Generator(), PS.ActivePower(), nlp.network_cache)
    return conv
end

function objective(nlp::ReducedSpaceEvaluator, u)
    # cost = cost_production(nlp.model, nlp.x, u, nlp.p)
    cost = cost_production(nlp.model, nlp.network_cache.pg)
    # TODO: determine if we should include λ' * g(x, u), even if ≈ 0
    return cost
end

# Private function to compute adjoint (should be inlined)
function _adjoint(λ, J, y)
    λ .= - J' \ y
end
function _adjoint(λ, J::CuSparseMatrixCSR{T}, y::CuVector{T}) where T
    # TODO: we SHOULD find a most efficient implementation
    Jt = CuArray(J') |> sparse
    return CUSOLVER.csrlsvqr!(Jt, -y, λ, 1e-8, one(Cint), 'O')
end

function gradient!(nlp::ReducedSpaceEvaluator, g, u)
    cache = nlp.network_cache
    xₖ = nlp.x
    ∇gₓ = nlp.ad.Jgₓ.J
    # Evaluate Jacobian of power flow equation on current u
    ∇gᵤ = jacobian(nlp.model, nlp.ad.Jgᵤ, cache)
    ∇fₓ, ∇fᵤ = cost_production_adjoint(nlp.model, cache)
    # Update adjoint
    λₖ = nlp.λ
    _adjoint(λₖ, ∇gₓ, ∇fₓ)
    # compute inplace reduced gradient (g = ∇fᵤ + (∇gᵤ')*λₖ)
    copy!(g, ∇fᵤ)
    mul!(g, ∇gᵤ', λₖ, 1.0, 1.0)
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

function jacobian!(nlp::ReducedSpaceEvaluator, jac, u)
    xₖ = nlp.x
    ∇gₓ = nlp.ad.Jgₓ.J
    ∇gᵤ = nlp.ad.Jgᵤ.J
    nₓ = length(xₖ)
    MT = nlp.model.AT
    cnt = 1
    λ = nlp.λ
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
            _adjoint(λ, ∇gₓ, rhs)
            jac[cnt, :] .= Jᵤ[ix, :] + ∇gᵤ' * λ
            cnt += 1
        end
    end
end
