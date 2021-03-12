
"""
    ReducedSpaceEvaluator{T, VI, VT, MT} <: AbstractNLPEvaluator

Evaluator working in the reduced space corresponding to the
control variable `u`. Once a new point `u` is passed to the evaluator,
the user needs to call the method `update!` to find the corresponding
state `x(u)` satisfying the balance equation `g(x(u), u) = 0`.

Taking as input a formulation given as a `PolarForm` structure, the reduced evaluator
builds the bounds corresponding to the control `u`,
and initiate an `AutoDiffFactory` tailored to the problem. The reduced evaluator
could be instantiated on the host memory, or on a specific device (currently,
only CUDA is supported).

## Note
Mathematically, we set apart the state `x` from the control `u`,
and use a third variable `y` --- the by-product --- to store the remaining
values of the network.
In the implementation of `ReducedSpaceEvaluator`,
we only deal with a control `u` and an attribute `buffer`,
storing all the physical values needed to describe the network.
The attribute `buffer` stores the values of the control `u`, the state `x`
and the by-product `y`. Each time we are calling the method `update!`,
the values of the control are copied to the buffer.
The algorithm use only the physical representation of the network, more
convenient to use.

"""
mutable struct ReducedSpaceEvaluator{T, VI, VT, MT, Jacx, Jacu, JacCons, Hess} <: AbstractNLPEvaluator
    model::PolarForm{T, VI, VT, MT}
    λ::VT

    u_min::VT
    u_max::VT

    constraints::Vector{Function}
    g_min::VT
    g_max::VT

    # Cache
    buffer::PolarNetworkState{VI, VT}
    # AutoDiff
    state_jacobian::FullSpaceJacobian{Jacx, Jacu}
    obj_stack::AdjointStackObjective{VT} # / objective
    constraint_stacks::Vector{AdjointPolar{VT}} # / constraints
    constraint_jacobians::JacCons
    hessians::Hess

    # Options
    linear_solver::LinearSolvers.AbstractLinearSolver
    powerflow_solver::AbstractNonLinearSolver
    factorization::Union{Nothing, Factorization}
end

function ReducedSpaceEvaluator(
    model, x, u;
    constraints=Function[voltage_magnitude_constraints, active_power_constraints, reactive_power_constraints],
    linear_solver=DirectSolver(),
    powerflow_solver=NewtonRaphson(tol=1e-12),
    want_jacobian=true,
    want_hessian=true,
)
    # First, build up a network buffer
    buffer = get(model, PhysicalState())
    # Populate buffer with default values of the network, as stored
    # inside model
    init_buffer!(model, buffer)

    u_min, u_max = bounds(model, Control())
    λ = similar(x)

    MT = array_type(model)
    g_min = MT{eltype(x), 1}()
    g_max = MT{eltype(x), 1}()
    for cons in constraints
        cb, cu = bounds(model, cons)
        append!(g_min, cb)
        append!(g_max, cu)
    end

    obj_ad = AdjointStackObjective(model)
    state_ad = FullSpaceJacobian(model, power_balance)
    VT = typeof(g_min)
    cons_ad = AdjointPolar{VT}[]
    for cons in constraints
        push!(cons_ad, AdjointPolar(model))
    end

    # Jacobians
    cons_jac = nothing
    if want_jacobian
        cons_jac = ConstraintsJacobianStorage(model, constraints)
    end

    # Hessians
    hess_ad = nothing
    if want_hessian
        hess_ad = HessianStorage(model, constraints)
    end

    return ReducedSpaceEvaluator(
        model, λ, u_min, u_max,
        constraints, g_min, g_max,
        buffer,
        state_ad, obj_ad, cons_ad, cons_jac, hess_ad,
        linear_solver, powerflow_solver, nothing
    )
end
function ReducedSpaceEvaluator(
    datafile;
    device=KA.CPU(),
    options...
)
    # Load problem.
    pf = PS.PowerNetwork(datafile)
    polar = PolarForm(pf, device)

    x0 = initial(polar, State())
    uk = initial(polar, Control())

    return ReducedSpaceEvaluator(
        polar, x0, uk; options...
    )
end

array_type(nlp::ReducedSpaceEvaluator) = array_type(nlp.model)

n_variables(nlp::ReducedSpaceEvaluator) = length(nlp.u_min)
n_constraints(nlp::ReducedSpaceEvaluator) = length(nlp.g_min)

constraints_type(::ReducedSpaceEvaluator) = :inequality
has_hessian(::ReducedSpaceEvaluator) = true

# Getters
get(nlp::ReducedSpaceEvaluator, ::Constraints) = nlp.constraints
function get(nlp::ReducedSpaceEvaluator, ::State)
    x = similar(nlp.λ) ; fill!(x, 0)
    get!(nlp.model, State(), x, nlp.buffer)
    return x
end
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
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.ActiveLoad, values)
    setvalues!(nlp.model, attr, values)
    setvalues!(nlp.buffer, attr, values)
end
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.ReactiveLoad, values)
    setvalues!(nlp.model, attr, values)
    setvalues!(nlp.buffer, attr, values)
end

# Transfer network values inside buffer
function transfer!(
    nlp::ReducedSpaceEvaluator, vm, va, pg, qg,
)
    setvalues!(nlp.buffer, PS.VoltageMagnitude(), vm)
    setvalues!(nlp.buffer, PS.VoltageAngle(), va)
    setvalues!(nlp.buffer, PS.ActivePower(), pg)
    setvalues!(nlp.buffer, PS.ReactivePower(), qg)
end

# Initial position
function initial(nlp::ReducedSpaceEvaluator)
    u = similar(nlp.u_min) ; fill!(u, 0.0)
    return get!(nlp.model, Control(), u, nlp.buffer)
end

# Bounds
bounds(nlp::ReducedSpaceEvaluator, ::Variables) = (nlp.u_min, nlp.u_max)
bounds(nlp::ReducedSpaceEvaluator, ::Constraints) = (nlp.g_min, nlp.g_max)

function update!(nlp::ReducedSpaceEvaluator, u)
    jac_x = nlp.state_jacobian.x
    # Transfer control u into the network cache
    transfer!(nlp.model, nlp.buffer, u)
    # Get corresponding point on the manifold
    conv = powerflow(
        nlp.model,
        jac_x,
        nlp.buffer,
        nlp.powerflow_solver;
        solver=nlp.linear_solver
    )

    if !conv.has_converged
        error("Newton-Raphson algorithm failed to converge ($(conv.norm_residuals))")
        return conv
    end

    # Refresh values of active and reactive powers at generators
    update!(nlp.model, PS.Generators(), PS.ActivePower(), nlp.buffer)
    # Evaluate Jacobian of power flow equation on current u
    AutoDiff.jacobian!(nlp.model, nlp.state_jacobian.u, nlp.buffer)
    return conv
end

# TODO: determine if we should include λ' * g(x, u), even if ≈ 0
function objective(nlp::ReducedSpaceEvaluator, u)
    # Take as input the current cache, updated previously in `update!`.
    return objective(nlp.model, nlp.buffer)
end

function constraint!(nlp::ReducedSpaceEvaluator, g, u)
    ϕ = nlp.buffer
    mf = 1::Int
    mt = 0::Int
    for cons in nlp.constraints
        m_ = size_constraint(nlp.model, cons)::Int
        mt += m_
        cons_ = @view(g[mf:mt])
        cons(nlp.model, cons_, ϕ)
        mf += m_
    end
end


###
# First-order code
####
#
# compute inplace reduced gradient (g = ∇fᵤ + (∇gᵤ')*λ)
# equivalent to: g = ∇fᵤ - (∇gᵤ')*λ_neg
# (take λₖ_neg to avoid computing an intermediate array)
function reduced_gradient!(
    nlp::ReducedSpaceEvaluator, grad, ∂fₓ, ∂fᵤ, λ, u,
)
    ∇gᵤ = nlp.state_jacobian.u.J
    ∇gₓ = nlp.state_jacobian.x.J

    if isa(nlp.linear_solver, LinearSolvers.AbstractIterativeLinearSolver)
        # Iterative solver case
        ∇gT = LinearSolvers.get_transpose(nlp.linear_solver, ∇gₓ)
        # Switch preconditioner to transpose mode
        LinearSolvers.update!(nlp.linear_solver, ∇gT)
        # Compute adjoint and store value inside λₖ
        LinearSolvers.ldiv!(nlp.linear_solver, λ, ∇gT, ∂fₓ)
    elseif isa(u, CUDA.CuArray)
        ∇gT = LinearSolvers.get_transpose(nlp.linear_solver, ∇gₓ)
        LinearSolvers.ldiv!(nlp.linear_solver, λ, ∇gT, ∂fₓ)
    else
        # Direct solver case
        ∇gf = factorize(∇gₓ)
        LinearSolvers.ldiv!(nlp.linear_solver, λ, ∇gf', ∂fₓ)
        # Store factorization
        nlp.factorization = ∇gf
    end

    grad .= ∂fᵤ
    mul!(grad, transpose(∇gᵤ), λ, -1.0, 1.0)
    return λ
end
function reduced_gradient!(nlp::ReducedSpaceEvaluator, grad, ∂fₓ, ∂fᵤ, u)
    reduced_gradient!(nlp::ReducedSpaceEvaluator, grad, ∂fₓ, ∂fᵤ, nlp.λ, u)
end

# Compute only full gradient wrt x and u
function full_gradient!(nlp::ReducedSpaceEvaluator, gx, gu, u)
    buffer = nlp.buffer
    ∂obj = nlp.obj_stack
    # Evaluate adjoint of cost function and update inplace AdjointStackObjective
    gradient_objective!(nlp.model, ∂obj, buffer)
    copyto!(gx, ∂obj.∇fₓ)
    copyto!(gu, ∂obj.∇fᵤ)
end

function gradient!(nlp::ReducedSpaceEvaluator, g, u)
    buffer = nlp.buffer
    # Evaluate adjoint of cost function and update inplace AdjointStackObjective
    gradient_objective!(nlp.model, nlp.obj_stack, buffer)
    ∇fₓ, ∇fᵤ = nlp.obj_stack.∇fₓ, nlp.obj_stack.∇fᵤ

    reduced_gradient!(nlp, g, ∇fₓ, ∇fᵤ, u)
    return
end

function jacobian_structure(nlp::ReducedSpaceEvaluator)
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
    ∇cons = nlp.constraint_jacobians
    update_full_jacobian!(nlp.model, ∇cons, nlp.buffer)
    Jx = ∇cons.Jx
    Ju = ∇cons.Ju
    m, nₓ = size(Jx)
    ∇gᵤ = nlp.state_jacobian.u.J
    ∇gₓ = nlp.state_jacobian.x.J
    # Compute factorization with UMFPACK
    ∇gfac = nlp.factorization
    # Compute adjoints all in once, using the same factorization
    μ = zeros(nₓ, m)
    ldiv!(μ, ∇gfac', Jx')
    # Compute reduced Jacobian
    copy!(jac, Ju)
    mul!(jac, μ', ∇gᵤ, -1.0, 1.0)
    return
end

function jprod!(nlp::ReducedSpaceEvaluator, jv, u, v)
    nᵤ = length(u)
    m  = n_constraints(nlp)
    @assert nᵤ == length(v)

    ∇cons = nlp.constraint_jacobians
    update_full_jacobian!(nlp.model, ∇cons, nlp.buffer)
    Jx = ∇cons.Jx
    Ju = ∇cons.Ju

    ∇gᵤ = nlp.state_jacobian.u.J
    ∇gfac = nlp.factorization
    z = -(∇gfac \ (∇gᵤ * v))

    jv .= Ju * v .+ Jx * z
    return
end

function full_jtprod!(nlp::ReducedSpaceEvaluator, jvx, jvu, u, v)
    ∂obj = nlp.obj_stack
    fr_ = 0::Int
    for (cons, stack) in zip(nlp.constraints, nlp.constraint_stacks)
        n = size_constraint(nlp.model, cons)::Int
        mask = fr_+1:fr_+n
        vv = @view v[mask]
        # Compute jtprod of current constraint
        jtprod!(nlp.model, cons, stack, nlp.buffer, vv)
        jvx .+= stack.∂x
        jvu .+= stack.∂u
        fr_ += n
    end
end

function jtprod!(nlp::ReducedSpaceEvaluator, jv, u, v)
    ∂obj = nlp.obj_stack
    jvx = ∂obj.jvₓ ; fill!(jvx, 0)
    jvu = ∂obj.jvᵤ ; fill!(jvu, 0)
    full_jtprod!(nlp, jvx, jvu, u, v)
    reduced_gradient!(nlp, jv, jvx, jvu, jvx, u)
end

function ojtprod!(nlp::ReducedSpaceEvaluator, jv, u, σ, v)
    ∂obj = nlp.obj_stack
    jvx = ∂obj.jvₓ ; fill!(jvx, 0)
    jvu = ∂obj.jvᵤ ; fill!(jvu, 0)
    # compute gradient of objective
    full_gradient!(nlp, jvx, jvu, u)
    jvx .*= σ
    jvu .*= σ
    # compute transpose Jacobian vector product of constraints
    full_jtprod!(nlp, jvx, jvu, u, v)
    # Evaluate gradient in reduced space
    reduced_gradient!(nlp, jv, jvx, jvu, u)
    return
end

###
# Second-order code
####
# z = -(∇gₓ  \ (∇gᵤ * w))
function _second_order_adjoint_z!(
    nlp::ReducedSpaceEvaluator, z, w,
)
    ∇gᵤ = nlp.state_jacobian.u.J
    mul!(z, ∇gᵤ, w, -1.0, 0.0)

    if isa(z, CUDA.CuArray)
        ∇gₓ = nlp.state_jacobian.x.J
        LinearSolvers.ldiv!(nlp.linear_solver, z, ∇gₓ, z)
    else
        ∇gₓ = nlp.factorization
        ldiv!(∇gₓ, z)
    end
end

# ψ = -(∇gₓ' \ (∇²fₓₓ .+ ∇²gₓₓ))
function _second_order_adjoint_ψ!(
    nlp::ReducedSpaceEvaluator, ψ, ∂fₓ,
)
    if isa(ψ, CUDA.CuArray)
        ∇gₓ = nlp.state_jacobian.x.J
        ∇gT = LinearSolvers.get_transpose(nlp.linear_solver, ∇gₓ)
        LinearSolvers.ldiv!(nlp.linear_solver, ψ, ∇gT, ∂fₓ)
    else
        ∇gₓ = nlp.factorization
        ldiv!(ψ, ∇gₓ', ∂fₓ)
    end
end

function _reduced_hessian_prod!(
    nlp::ReducedSpaceEvaluator, hessvec, ∂fₓ, ∂fᵤ, tgt,
)
    nx = get(nlp.model, NumberOfState())
    nu = get(nlp.model, NumberOfControl())
    H = nlp.hessians
    ∇gᵤ = nlp.state_jacobian.u.J
    λ = - nlp.λ
    ψ = H.ψ

    hv = H.tmp_hv

    ## POWER BALANCE HESSIAN
    AutoDiff.adj_hessian_prod!(nlp.model, H.state, hv, nlp.buffer, λ, tgt)
    ∂fₓ .+= hv[1:nx]
    ∂fᵤ .+= hv[nx+1:nx+nu]

    # Second order adjoint
    _second_order_adjoint_ψ!(nlp, ψ, ∂fₓ)

    hessvec .+= ∂fᵤ
    mul!(hessvec, transpose(∇gᵤ), ψ, -1.0, 1.0)
    return
end

function hessprod!(nlp::ReducedSpaceEvaluator, hessvec, u, w)
    @assert nlp.hessians != nothing

    nx = get(nlp.model, NumberOfState())
    nu = get(nlp.model, NumberOfControl())
    buffer = nlp.buffer
    H = nlp.hessians

    fill!(hessvec, 0.0)

    # Two vector products
    tgt = H.tmp_tgt
    hv = H.tmp_hv
    z = H.z

    _second_order_adjoint_z!(nlp, z, w)

    # Init tangent
    tgt[1:nx] .= z
    tgt[1+nx:nx+nu] .= w

    ∂f = similar(buffer.pg)
    ∂²f = similar(buffer.pg)

    # Adjoint and Hessian of cost function
    adjoint_cost!(nlp.model, ∂f, buffer.pg)
    hessian_cost!(nlp.model, ∂²f)

    ## OBJECTIVE HESSIAN
    hessian_prod_objective!(nlp.model, H.obj, nlp.obj_stack, hv, ∂²f, ∂f, buffer, tgt)
    ∇²fx = hv[1:nx]
    ∇²fu = hv[nx+1:nx+nu]

    _reduced_hessian_prod!(nlp, hessvec, ∇²fx, ∇²fu, tgt)

    return
end

function hessian_lagrangian_penalty_prod!(
    nlp::ReducedSpaceEvaluator, hessvec, u, y, σ, w, D,
)
    @assert nlp.hessians != nothing

    nx = get(nlp.model, NumberOfState())
    nu = get(nlp.model, NumberOfControl())
    buffer = nlp.buffer
    H = nlp.hessians

    fill!(hessvec, 0.0)

    z = H.z
    _second_order_adjoint_z!(nlp, z, w)

    # Two vector products
    tgt = H.tmp_tgt
    hv = H.tmp_hv

    # Init tangent
    tgt[1:nx] .= z
    tgt[1+nx:nx+nu] .= w

    ∂f = similar(buffer.pg)
    ∂²f = similar(buffer.pg)
    # Adjoint and Hessian of cost function
    adjoint_cost!(nlp.model, ∂f, buffer.pg)
    hessian_cost!(nlp.model, ∂²f)

    ## OBJECTIVE HESSIAN
    hessian_prod_objective!(nlp.model, H.obj, nlp.obj_stack, hv, ∂²f, ∂f, buffer, tgt)
    ∇²Lx = σ .* hv[1:nx]
    ∇²Lu = σ .* hv[nx+1:nx+nu]

    # CONSTRAINT HESSIAN
    shift = 0
    for (cons, Hc) in zip(nlp.constraints, H.constraints)
        m = size_constraint(nlp.model, cons)::Int
        mask = shift+1:shift+m
        yc = @view y[mask]
        AutoDiff.adj_hessian_prod!(nlp.model, Hc, hv, buffer, yc, tgt)
        ∇²Lx .+= hv[1:nx]
        ∇²Lu .+= hv[nx+1:nx+nu]
        shift += m
    end
    # Add Hessian of quadratic penalty
    if !iszero(D)
        update_full_jacobian!(nlp.model, nlp.constraint_jacobians, buffer)
        Jx = nlp.constraint_jacobians.Jx
        Ju = nlp.constraint_jacobians.Ju
        DD = Diagonal(D)
        ∇²Lx .+= Jx' * (DD * (Jx * z)) .+ Jx' * (DD * (Ju * w))
        ∇²Lu .+= Ju' * (DD * (Jx * z)) .+ Ju' * (DD * (Ju * w))
    end

    # Second order adjoint
    _reduced_hessian_prod!(nlp, hessvec, ∇²Lx, ∇²Lu, tgt)

    return
end

# Return lower-triangular matrix
function hessian_structure(nlp::ReducedSpaceEvaluator)
    n = n_variables(nlp)
    rows = Int[r for r in 1:n for c in 1:r]
    cols = Int[c for r in 1:n for c in 1:r]
    return rows, cols
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
    # Reset buffer
    init_buffer!(nlp.model, nlp.buffer)
    return
end

