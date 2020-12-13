
"""
    ReducedSpaceEvaluator{T} <: AbstractNLPEvaluator

Evaluator working in the reduced space corresponding to the
control variable `u`. Once a new point `u` is passed to the evaluator,
the user needs to call the method `update!` to find the corresponding
state `x(u)` satisfying the equilibrium equation `g(x(u), u) = 0`.

Taking as input a given `AbstractFormulation`, the reduced evaluator
builds the bounds corresponding to the control `u` and the state `x`,
and initiate an `AutoDiffFactory` tailored to the problem. The reduced evaluator
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
    autodiff::AutoDiffFactory
    ∇gᵗ::AbstractMatrix
    linear_solver::LinearSolvers.AbstractLinearSolver
    ε_tol::Float64
end

function ReducedSpaceEvaluator(
    model, x, u, p;
    constraints=Function[state_constraint, power_constraints],
    linear_solver=DirectSolver(),
    ε_tol=1e-12,
    verbose_level=VERBOSE_LEVEL_NONE,
)
    # First, build up a network buffer
    buffer = get(model, PhysicalState())
    # Initiate adjoint
    λ = similar(x)
    # Build up AutoDiff factory
    jx, ju, adjoint_f = init_autodiff_factory(model, buffer)
    ad = AutoDiffFactory(jx, ju, adjoint_f)

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
function ReducedSpaceEvaluator(
    datafile;
    device=CPU(),
    options...
)
    # Load problem.
    pf = PS.PowerNetwork(datafile)
    polar = PolarForm(pf, device)

    x0 = initial(polar, State())
    p = initial(polar, Parameters())
    uk = initial(polar, Control())

    return ReducedSpaceEvaluator(
        polar, x0, uk, p; options...
    )
end

# TODO: add constructor from PowerNetwork

type_array(nlp::ReducedSpaceEvaluator) = typeof(nlp.u_min)

n_variables(nlp::ReducedSpaceEvaluator) = length(nlp.u_min)
n_constraints(nlp::ReducedSpaceEvaluator) = length(nlp.g_min)

# Getters
get(nlp::ReducedSpaceEvaluator, ::Constraints) = nlp.constraints
get(nlp::ReducedSpaceEvaluator, ::State) = nlp.x
get(nlp::ReducedSpaceEvaluator, ::PhysicalState) = nlp.buffer
get(nlp::ReducedSpaceEvaluator, ::AutoDiffBackend) = nlp.autodiff
# Physics
get(nlp::ReducedSpaceEvaluator, ::PS.VoltageMagnitude) = nlp.buffer.vmag
get(nlp::ReducedSpaceEvaluator, ::PS.VoltageAngle) = nlp.buffer.vang
get(nlp::ReducedSpaceEvaluator, ::PS.ActivePower) = nlp.buffer.pg
get(nlp::ReducedSpaceEvaluator, ::PS.ReactivePower) = nlp.buffer.qg
function get(nlp::ReducedSpaceEvaluator, attr::PS.AbstractNetworkAttribute)
    return get(nlp.model, attr)
end

# Setters
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.AbstractNetworkValues, values)
    setvalues!(nlp.model, attr, values)
end

# Initial position
initial(nlp::ReducedSpaceEvaluator) = initial(nlp.model, Control())

# Bounds
bounds(nlp::ReducedSpaceEvaluator, ::Variables) = nlp.u_min, nlp.u_max
bounds(nlp::ReducedSpaceEvaluator, ::Constraints) = nlp.g_min, nlp.g_max

function update!(nlp::ReducedSpaceEvaluator, u; verbose_level=0)
    x₀ = nlp.x
    jac_x = nlp.autodiff.Jgₓ
    # Transfer x, u, p into the network cache
    transfer!(nlp.model, nlp.buffer, nlp.x, u)
    # Get corresponding point on the manifold
    conv = powerflow(nlp.model, jac_x, nlp.buffer, tol=nlp.ε_tol;
                     solver=nlp.linear_solver, verbose_level=verbose_level)
    if !conv.has_converged
        error("Newton-Raphson algorithm failed to converge ($(conv.norm_residuals))")
        return conv
    end
    ∇gₓ = nlp.autodiff.Jgₓ.J
    nlp.∇gᵗ = LinearSolvers.get_transpose(nlp.linear_solver, ∇gₓ)
    # Switch preconditioner to transpose mode
    if isa(nlp.linear_solver, LinearSolvers.AbstractIterativeLinearSolver)
        LinearSolvers.update!(nlp.linear_solver, nlp.∇gᵗ)
    end

    # Update value of nlp.x with new network state
    get!(nlp.model, State(), nlp.x, nlp.buffer)
    # Refresh value of the active power of the generators
    refresh!(nlp.model, PS.Generator(), PS.ActivePower(), nlp.buffer)
    return conv
end

function objective(nlp::ReducedSpaceEvaluator, u)
    # Take as input the current cache, updated previously in `update!`.
    pg = get(nlp, PS.ActivePower())
    cost = cost_production(nlp.model, pg)
    # TODO: determine if we should include λ' * g(x, u), even if ≈ 0
    return cost
end

function update_jacobian!(nlp::ReducedSpaceEvaluator, ::Control)
    jacobian(nlp.model, nlp.autodiff.Jgᵤ, nlp.buffer)
end

# compute inplace reduced gradient (g = ∇fᵤ + (∇gᵤ')*λₖ)
# equivalent to: g = ∇fᵤ - (∇gᵤ')*λₖ_neg
# (take λₖ_neg to avoid computing an intermediate array)
function reduced_gradient!(
    nlp::ReducedSpaceEvaluator, grad, ∂fₓ, ∂fᵤ,
)
    λₖ = nlp.λ
    ∇gᵤ = nlp.autodiff.Jgᵤ.J
    # Compute adjoint and store value inside λₖ
    LinearSolvers.ldiv!(nlp.linear_solver, λₖ, nlp.∇gᵗ, ∂fₓ)
    grad .= ∂fᵤ
    mul!(grad, transpose(∇gᵤ), λₖ, -1.0, 1.0)
end

function gradient!(nlp::ReducedSpaceEvaluator, g, u)
    buffer = nlp.buffer
    # Evaluate adjoint of cost function and update inplace AdjointStackObjective
    ∂cost(nlp.model, nlp.autodiff.∇f, buffer)
    ∇fₓ, ∇fᵤ = nlp.autodiff.∇f.∇fₓ, nlp.autodiff.∇f.∇fᵤ

    # Evaluate Jacobian of power flow equation on current u
    update_jacobian!(nlp, Control())
    reduced_gradient!(nlp, g, ∇fₓ, ∇fᵤ)
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

function jacobian_structure(nlp::ReducedSpaceEvaluator)
    S = type_array(nlp)
    m, n = n_constraints(nlp), n_variables(nlp)
    nnzj = m * n
    rows = zeros(Int, nnzj)
    cols = zeros(Int, nnzj)
    jacobian_structure!(nlp, rows, cols)
    return rows, cols
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
    ∇gₓ = nlp.autodiff.Jgₓ.J
    ∇gᵤ = nlp.autodiff.Jgᵤ.J
    nₓ = length(xₖ)
    MT = nlp.model.AT
    μ = similar(nlp.λ)
    ∂obj = nlp.autodiff.∇f
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
    ∂obj = nlp.autodiff.∇f
    # Get adjoint
    jtprod(model, cons, ∂obj, nlp.buffer, v)
    jvx, jvu = ∂obj.∇fₓ, ∂obj.∇fᵤ
    reduced_gradient!(nlp, jv, jvx, jvu)
end

function jtprod!(nlp::ReducedSpaceEvaluator, jv, u, v)
    ∂obj = nlp.autodiff.∇f
    jvx = ∂obj.jvₓ
    jvu = ∂obj.jvᵤ
    fill!(jvx, 0)
    fill!(jvu, 0)

    fr_ = 0
    for cons in nlp.constraints
        n = size_constraint(nlp.model, cons)
        mask = fr_+1:fr_+n
        vv = @view v[mask]
        # Compute jtprod of current constraint
        jtprod(nlp.model, cons, ∂obj, nlp.buffer, vv)
        jvx .+= ∂obj.∇fₓ
        jvu .+= ∂obj.∇fᵤ
        fr_ += n
    end
    reduced_gradient!(nlp, jv, jvx, jvu)
    return
end

# Utils function
function primal_infeasibility!(nlp::ReducedSpaceEvaluator, cons, u)
    constraint!(nlp, cons, u) # Evaluate constraints
    (n_inf, err_inf, n_sup, err_sup) = _check(cons, nlp.g_min, nlp.g_max)
    return max(err_inf, err_sup)
end
function primal_infeasibility(nlp::ReducedSpaceEvaluator, u)
    cons = similar(nlp.g_min) ; fill!(cons, 0)
    return primal_infeasibility!(nlp, cons, u)
end

# Printing
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

function Base.show(io::IO, nlp::ReducedSpaceEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    println(io, "A ReducedSpaceEvaluator object")
    println(io, "    * device: ", nlp.model.device)
    println(io, "    * #vars: ", n)
    println(io, "    * #cons: ", m)
    println(io, "    * constraints:")
    for cons in nlp.constraints
        println(io, "        - ", cons)
    end
    print(io, "    * linear solver: ", nlp.linear_solver)
end

function reset!(nlp::ReducedSpaceEvaluator)
    # Reset adjoint
    fill!(nlp.λ, 0)
    # Reset initial state
    copy!(nlp.x, initial(nlp.model, State()))
    # Reset buffer
    nlp.buffer = get(nlp.model, PhysicalState())
end

