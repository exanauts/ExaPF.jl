
"""
    ReducedSpaceEvaluator{T} <: AbstractNLPEvaluator

Evaluator working in the reduced space corresponding to the
control variable `u`. Once a new point `u` is passed to the evaluator,
the user needs to call the method `update!` to find the corresponding
state `x(u)` satisfying the equilibrium equation `g(x(u), u) = 0`.

Taking as input a given `AbstractFormulation`, the reduced evaluator
builds the bounds corresponding to the control `u` and the state `x`,
and initiate an `ADFactory` tailored to the problem. The reduced evaluator
could be instantiated on the main memory, or on a specific device (currently,
only CUDA is supported).

"""
mutable struct ReducedSpaceEvaluator{T} <: AbstractNLPEvaluator
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

    buffer::AbstractNetworkBuffer
    ad::ADFactory
    ∇gᵗ::AbstractMatrix
    linear_solver::LinearSolvers.AbstractLinearSolver
    ε_tol::Float64
end

function ReducedSpaceEvaluator(model, x, u, p;
                               constraints=Function[state_constraint],
                               ε_tol=1e-12, linear_solver=DirectSolver(), npartitions=2,
                               verbose_level=VERBOSE_LEVEL_NONE)
    # First, build up a network buffer
    buffer = get(model, PhysicalState())
    # Initiate adjoint
    λ = similar(x)
    # Build up AD factory
    jx, ju, adjoint_f = init_ad_factory(model, buffer)
    ad = ADFactory(jx, ju, adjoint_f)

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
                                 buffer,
                                 ad, jx.J, linear_solver, ε_tol)
end

n_variables(nlp::ReducedSpaceEvaluator) = length(nlp.u_min)
n_constraints(nlp::ReducedSpaceEvaluator) = length(nlp.g_min)

function update!(nlp::ReducedSpaceEvaluator, u; verbose_level=0)
    x₀ = nlp.x
    jac_x = nlp.ad.Jgₓ
    # Transfer x, u, p into the network cache
    transfer!(nlp.model, nlp.buffer, nlp.x, u, nlp.p)
    # Get corresponding point on the manifold
    conv = powerflow(nlp.model, jac_x, nlp.buffer, tol=nlp.ε_tol;
                     solver=nlp.linear_solver, verbose_level=verbose_level)
    if !conv.has_converged
        println(conv.norm_residuals)
        error("Failure")
    end
    ∇gₓ = nlp.ad.Jgₓ.J
    nlp.∇gᵗ = LinearSolvers.get_transpose(nlp.linear_solver, ∇gₓ)

    # Update value of nlp.x with new network state
    get!(nlp.model, State(), nlp.x, nlp.buffer)
    # Refresh value of the active power of the generators
    refresh!(nlp.model, PS.Generator(), PS.ActivePower(), nlp.buffer)
    return conv
end

function objective(nlp::ReducedSpaceEvaluator, u)
    # Take as input the current cache, updated previously in `update!`.
    cost = cost_production(nlp.model, nlp.buffer.pg)
    # TODO: determine if we should include λ' * g(x, u), even if ≈ 0
    return cost
end

# compute inplace reduced gradient (g = ∇fᵤ + (∇gᵤ')*λₖ)
# equivalent to: g = ∇fᵤ - (∇gᵤ')*λₖ_neg
# (take λₖ_neg to avoid computing an intermediate array)
function _reduced_gradient!(g, ∇fᵤ, ∇gᵤ, λₖ_neg)
    g .= ∇fᵤ
    mul!(g, transpose(∇gᵤ), λₖ_neg, -1.0, 1.0)
end

function gradient!(nlp::ReducedSpaceEvaluator, g, u)
    buffer = nlp.buffer
    xₖ = nlp.x
    ∇gₓ = nlp.ad.Jgₓ.J
    # Evaluate Jacobian of power flow equation on current u
    ∇gᵤ = jacobian(nlp.model, nlp.ad.Jgᵤ, buffer)
    # Evaluate adjoint of cost function and update inplace ObjectiveAD
    cost_production_adjoint(nlp.model, nlp.ad.∇f, buffer)

    ∇fₓ, ∇fᵤ = nlp.ad.∇f.∇fₓ, nlp.ad.∇f.∇fᵤ
    # Update (negative) adjoint
    λₖ_neg = nlp.λ
    LinearSolvers.ldiv!(nlp.linear_solver, λₖ_neg, nlp.∇gᵗ, ∇fₓ)
    _reduced_gradient!(g, ∇fᵤ, ∇gᵤ, λₖ_neg)
    return nothing
end

function constraint!(nlp::ReducedSpaceEvaluator, g, u)
    xₖ = nlp.x
    ϕ = nlp.buffer
    # First: state constraint
    mf = 1
    mt = 0
    for cons in nlp.constraints
        m_ = size_constraint(nlp.model, cons)
        mt += m_
        cons_ = @view(g[mf:mt])
        cons(nlp.model, cons_, ϕ)
        mf += m_
    end
end

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
    model = nlp.model
    xₖ = nlp.x
    ∇gₓ = nlp.ad.Jgₓ.J
    ∇gᵤ = nlp.ad.Jgᵤ.J
    nₓ = length(xₖ)
    MT = nlp.model.AT
    μ = similar(nlp.λ)
    ∂obj = nlp.ad.∇f
    cnt = 1

    for cons in nlp.constraints
        mc_ = size_constraint(nlp.model, cons)
        for i_cons in 1:mc_
            jacobian(model, cons, i_cons, ∂obj, nlp.buffer)
            jx, ju = ∂obj.∇fₓ, ∂obj.∇fᵤ
            # Get adjoint
            LinearSolvers.ldiv!(nlp.linear_solver, μ, nlp.∇gᵗ, jx)
            jac[cnt, :] .= (ju .- ∇gᵤ' * μ)
            cnt += 1
        end
    end
end

function jtprod!(nlp::ReducedSpaceEvaluator, cons, jv, u, v; start=1)
    model = nlp.model
    xₖ = nlp.x
    ∇gₓ = nlp.ad.Jgₓ.J
    ∇gᵤ = nlp.ad.Jgᵤ.J
    nₓ = length(xₖ)
    μ = nlp.λ

    ∂obj = nlp.ad.∇f
    # Get adjoint
    jtprod(model, cons, ∂obj, nlp.buffer, v)
    jvx, jvu = ∂obj.∇fₓ, ∂obj.∇fᵤ
    # jv .+= (ju .- ∇gᵤ' * μ)
    LinearSolvers.ldiv!(nlp.linear_solver, μ, nlp.∇gᵗ, jvx)
    jv .+= jvu
    mul!(jv, transpose(∇gᵤ), μ, -1.0, 1.0)
end
function jtprod!(nlp::ReducedSpaceEvaluator, jv, u, v)
    fr_ = 0
    for cons in nlp.constraints
        n = size_constraint(nlp.model, cons)
        mask = fr_+1:fr_+n
        vv = @view v[mask]
        jtprod!(nlp, cons, jv, u, vv)
        fr_ += n
    end
end

# Utils function

function sanity_check(nlp::ReducedSpaceEvaluator, u, cons)
    println("Check violation of constraints")
    print("Control  \t")
    (n_inf, err_inf, n_sup, err_sup) = _check(u, nlp.u_min, nlp.u_max)
    @printf("UB: %.4e (%d)    LB: %.4e (%d)\n",
            err_sup, n_sup, err_inf, n_inf)
    print("State    \t")
    (n_inf, err_inf, n_sup, err_sup) = _check(nlp.x, nlp.x_min, nlp.x_max)
    @printf("UB: %.4e (%d)    LB: %.4e (%d)\n",
            err_sup, n_sup, err_inf, n_inf)
    print("Constraints\t")
    (n_inf, err_inf, n_sup, err_sup) = _check(cons, nlp.g_min, nlp.g_max)
    @printf("UB: %.4e (%d)    LB: %.4e (%d)\n",
            err_sup, n_sup, err_inf, n_inf)
end

