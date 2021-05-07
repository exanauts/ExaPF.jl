
"""
    ReducedSpaceEvaluator{T, VI, VT, MT, Jacx, Jacu, JacCons, Hess} <: AbstractNLPEvaluator

Reduced-space evaluator projecting the optimization problem
into the powerflow manifold defined by the nonlinear equation ``g(x, u) = 0``.
The state ``x`` is defined implicitly, as a function of the control
``u``. Hence, the powerflow equation is implicitly satisfied
when we are using this evaluator.

Once a new point `u` is passed to the evaluator,
the user needs to call the method `update!` to find the corresponding
state ``x(u)`` satisfying the balance equation ``g(x(u), u) = 0``.

Taking as input a [`PolarForm`](@ref) structure, the reduced evaluator
builds the bounds corresponding to the control `u`,
The reduced evaluator could be instantiated on the host memory, or on a specific device
(currently, only CUDA is supported).

## Examples

```julia-repl
julia> datafile = "case9.m"  # specify a path to a MATPOWER instance
julia> nlp = ReducedSpaceEvaluator(datafile)
A ReducedSpaceEvaluator object
    * device: KernelAbstractions.CPU()
    * #vars: 5
    * #cons: 10
    * constraints:
        - voltage_magnitude_constraints
        - active_power_constraints
        - reactive_power_constraints
    * linear solver: ExaPF.LinearSolvers.DirectSolver()
```

If a GPU is available, we could instantiate `nlp` as

```julia-repl
julia> nlp_gpu = ReducedSpaceEvaluator(datafile; device=CUDADevice())
A ReducedSpaceEvaluator object
    * device: KernelAbstractions.CUDADevice()
    * #vars: 5
    * #cons: 10
    * constraints:
        - voltage_magnitude_constraints
        - active_power_constraints
        - reactive_power_constraints
    * linear solver: ExaPF.LinearSolvers.DirectSolver()

```

## Note
Mathematically, we set apart the state ``x`` from the control ``u``,
and use a third variable ``y`` --- the by-product --- to denote the remaining
values of the network.
In the implementation of `ReducedSpaceEvaluator`,
we only deal with a control `u` and an attribute `buffer`,
storing all the physical values needed to describe the network.
The attribute `buffer` stores the values of the control `u`, the state `x`
and the by-product `y`. Each time we are calling the method `update!`,
the values of the control are copied into the buffer.

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
    obj_stack::AutoDiff.TapeMemory{typeof(active_power_generation), AdjointStackObjective{VT}, Nothing}
    cons_stacks::Vector{AutoDiff.TapeMemory} # / constraints
    constraint_jacobians::JacCons
    hessians::Hess

    # Options
    linear_solver::LinearSolvers.AbstractLinearSolver
    powerflow_solver::AbstractNonLinearSolver
    has_jacobian::Bool
    update_jacobian::Bool
    has_hessian::Bool
end

function ReducedSpaceEvaluator(
    model::PolarForm{T, VI, VT, MT};
    constraints=Function[voltage_magnitude_constraints, active_power_constraints, reactive_power_constraints],
    linear_solver=direct_linear_solver(model),
    powerflow_solver=NewtonRaphson(tol=1e-12),
    want_jacobian=true,
    want_hessian=true,
) where {T, VI, VT, MT}
    # First, build up a network buffer
    buffer = get(model, PhysicalState())
    # Populate buffer with default values of the network, as stored
    # inside model
    init_buffer!(model, buffer)

    u_min, u_max = bounds(model, Control())
    λ = similar(buffer.dx)

    g_min = VT()
    g_max = VT()
    for cons in constraints
        cb, cu = bounds(model, cons)
        append!(g_min, cb)
        append!(g_max, cu)
    end

    obj_ad = pullback_objective(model)
    state_ad = FullSpaceJacobian(model, power_balance)
    cons_ad = AutoDiff.TapeMemory[]
    for cons in constraints
        push!(cons_ad, AutoDiff.TapeMemory(model, cons, VT))
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
        linear_solver, powerflow_solver, want_jacobian, false, want_hessian,
    )
end
function ReducedSpaceEvaluator(datafile::String; device=KA.CPU(), options...)
    return ReducedSpaceEvaluator(PolarForm(datafile, device); options...)
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
get(nlp::ReducedSpaceEvaluator, ::PS.ActivePower) = nlp.buffer.pgen
get(nlp::ReducedSpaceEvaluator, ::PS.ReactivePower) = nlp.buffer.qgen
function get(nlp::ReducedSpaceEvaluator, attr::PS.AbstractNetworkAttribute)
    return get(nlp.model, attr)
end

# Setters
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.AbstractNetworkValues, values)
    setvalues!(nlp.model, attr, values)
end
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.ActiveLoad, values)
    setvalues!(nlp.buffer, attr, values)
end
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.ReactiveLoad, values)
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

## Callbacks
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
        linear_solver=nlp.linear_solver
    )

    if !conv.has_converged
        error("Newton-Raphson algorithm failed to converge ($(conv.norm_residuals))")
        return conv
    end

    # Refresh values of active and reactive powers at generators
    update!(nlp.model, PS.Generators(), PS.ActivePower(), nlp.buffer)
    # Evaluate Jacobian of power flow equation on current u
    AutoDiff.jacobian!(nlp.model, nlp.state_jacobian.u, nlp.buffer)
    # Specify that constraint's Jacobian is not up to date
    nlp.update_jacobian = nlp.has_jacobian
    return conv
end

# TODO: determine if we should include λ' * g(x, u), even if ≈ 0
function objective(nlp::ReducedSpaceEvaluator, u)
    # Take as input the current cache, updated previously in `update!`.
    return cost_production(nlp.model, nlp.buffer)
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


function _forward_solve!(nlp::ReducedSpaceEvaluator, y, x)
    if isa(y, CUDA.CuArray)
        ∇gₓ = nlp.state_jacobian.x.J
        LinearSolvers.ldiv!(nlp.linear_solver, y, ∇gₓ, x)
    else
        LinearSolvers.ldiv!(nlp.linear_solver, y, x)
    end
end

function _backward_solve!(nlp::ReducedSpaceEvaluator, y::VT, x::VT) where {VT <: AbstractArray}
    ∇gₓ = nlp.state_jacobian.x.J
    if isa(nlp.linear_solver, LinearSolvers.AbstractIterativeLinearSolver)
        # Iterative solver case
        ∇gT = LinearSolvers.get_transpose(nlp.linear_solver, ∇gₓ)
        # Switch preconditioner to transpose mode
        LinearSolvers.update!(nlp.linear_solver, ∇gT)
        # Compute adjoint and store value inside λₖ
        LinearSolvers.ldiv!(nlp.linear_solver, y, ∇gT, x)
    elseif isa(y, CUDA.CuArray)
        ∇gT = LinearSolvers.get_transpose(nlp.linear_solver, ∇gₓ)
        LinearSolvers.ldiv!(nlp.linear_solver, y, ∇gT, x)
    else
        LinearSolvers.rdiv!(nlp.linear_solver, y, x)
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

    # λ = ∇gₓ' \ ∂fₓ
    _backward_solve!(nlp, λ, ∂fₓ)

    grad .= ∂fᵤ
    mul!(grad, transpose(∇gᵤ), λ, -1.0, 1.0)
    return
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
    copyto!(gx, ∂obj.stack.∇fₓ)
    copyto!(gu, ∂obj.stack.∇fᵤ)
end

function gradient!(nlp::ReducedSpaceEvaluator, g, u)
    buffer = nlp.buffer
    # Evaluate adjoint of cost function and update inplace AdjointStackObjective
    gradient_objective!(nlp.model, nlp.obj_stack, buffer)
    ∇fₓ, ∇fᵤ = nlp.obj_stack.stack.∇fₓ, nlp.obj_stack.stack.∇fᵤ

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
    for i in 1:n # number of variables
        for c in 1:m #number of constraints
            rows[idx] = c ; cols[idx] = i
            idx += 1
        end
    end
end

function _update_full_jacobian_constraints!(nlp)
    if nlp.update_jacobian
        update_full_jacobian!(nlp.model, nlp.constraint_jacobians, nlp.buffer)
        nlp.update_jacobian = false
    end
end

# Works only on the CPU!
function jacobian!(nlp::ReducedSpaceEvaluator, J, u)
    m, n = n_constraints(nlp), n_variables(nlp)
    jac = reshape(J, m, n)

    _update_full_jacobian_constraints!(nlp)

    ∇cons = nlp.constraint_jacobians
    Jx = ∇cons.Jx
    Ju = ∇cons.Ju
    m, nₓ = size(Jx)
    m, nᵤ = size(Ju)
    ∇gᵤ = nlp.state_jacobian.u.J
    ∇gₓ = nlp.state_jacobian.x.J
    # Compute state sensitivities all in once
    μ = zeros(nₓ, nᵤ)
    LinearSolvers.ldiv!(nlp.linear_solver, μ, ∇gᵤ)
    # Compute reduced Jacobian
    copy!(jac, Ju)
    mul!(jac, Jx, μ, -1.0, 1.0)
    return
end

function jprod!(nlp::ReducedSpaceEvaluator, jv, u, v)
    nᵤ = length(u)
    m  = n_constraints(nlp)
    @assert nᵤ == length(v)

    _update_full_jacobian_constraints!(nlp)

    ∇cons = nlp.constraint_jacobians

    Jx = ∇cons.Jx
    Ju = ∇cons.Ju

    ∇gᵤ = nlp.state_jacobian.u.J
    rhs = nlp.buffer.dx
    z = nlp.buffer.balance
    # init RHS
    mul!(rhs, ∇gᵤ, v)
    # Compute z
    _forward_solve!(nlp, z, rhs)

    # jv .= Ju * v .- Jx * z
    mul!(jv, Ju, v)
    mul!(jv, Jx, z, -1.0, 1.0)
    return
end

function full_jtprod!(nlp::ReducedSpaceEvaluator, jvx, jvu, u, v)
    fr_ = 0::Int
    for (cons, stack) in zip(nlp.constraints, nlp.cons_stacks)
        n = size_constraint(nlp.model, cons)::Int
        mask = fr_+1:fr_+n
        vv = @view v[mask]
        # Compute jtprod of current constraint
        jacobian_transpose_product!(nlp.model, stack, nlp.buffer, vv)
        jvx .+= stack.stack.∂x
        jvu .+= stack.stack.∂u
        fr_ += n
    end
end

function jtprod!(nlp::ReducedSpaceEvaluator, jv, u, v)
    ∂obj = nlp.obj_stack
    μ = nlp.buffer.balance
    jvx = ∂obj.stack.jvₓ ; fill!(jvx, 0)
    jvu = ∂obj.stack.jvᵤ ; fill!(jvu, 0)
    full_jtprod!(nlp, jvx, jvu, u, v)
    reduced_gradient!(nlp, jv, jvx, jvu, μ, u)
end

function ojtprod!(nlp::ReducedSpaceEvaluator, jv, u, σ, v)
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
    rhs = nlp.buffer.dx
    mul!(rhs, ∇gᵤ, w, -1.0, 0.0)
    _forward_solve!(nlp, z, rhs)
end

# ψ = -(∇gₓ' \ (∇²fₓₓ .+ ∇²gₓₓ))
function _second_order_adjoint_ψ!(
    nlp::ReducedSpaceEvaluator, ψ, ∂fₓ,
)
    _backward_solve!(nlp, ψ, ∂fₓ)
end

function _reduced_hessian_prod!(
    nlp::ReducedSpaceEvaluator, hessvec, ∂fₓ, ∂fᵤ, tgt,
)
    nx = get(nlp.model, NumberOfState())
    nu = get(nlp.model, NumberOfControl())
    H = nlp.hessians
    ∇gᵤ = nlp.state_jacobian.u.J
    ψ = H.ψ

    hv = H.tmp_hv

    ## POWER BALANCE HESSIAN
    AutoDiff.adj_hessian_prod!(nlp.model, H.state, hv, nlp.buffer, nlp.λ, tgt)
    ∂fₓ .-= @view hv[1:nx]
    ∂fᵤ .-= @view hv[nx+1:nx+nu]

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
    copyto!(tgt, 1, z, 1, nx)
    copyto!(tgt, nx+1, w, 1, nu)

    ∂f = similar(buffer.pgen)
    ∂²f = similar(buffer.pgen)

    # Adjoint and Hessian of cost function
    adjoint_cost!(nlp.model, ∂f, buffer.pgen)
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

    ∂f = similar(buffer.pgen)
    ∂²f = similar(buffer.pgen)
    # Adjoint and Hessian of cost function
    adjoint_cost!(nlp.model, ∂f, buffer.pgen)
    hessian_cost!(nlp.model, ∂²f)

    ## OBJECTIVE HESSIAN
    hessian_prod_objective!(nlp.model, H.obj, nlp.obj_stack, hv, ∂²f, ∂f, buffer, tgt)
    ∇²Lx = σ .* @view hv[1:nx]
    ∇²Lu = σ .* @view hv[nx+1:nx+nu]

    # CONSTRAINT HESSIAN
    shift = 0
    hvx = @view hv[1:nx]
    hvu = @view hv[nx+1:nx+nu]
    for (cons, Hc) in zip(nlp.constraints, H.constraints)
        m = size_constraint(nlp.model, cons)::Int
        mask = shift+1:shift+m
        yc = @view y[mask]
        AutoDiff.adj_hessian_prod!(nlp.model, Hc, hv, buffer, yc, tgt)
        ∇²Lx .+= hvx
        ∇²Lu .+= hvu
        shift += m
    end
    # Add Hessian of quadratic penalty
    diagjac = similar(y)
    if !iszero(D)
        _update_full_jacobian_constraints!(nlp)
        Jx = nlp.constraint_jacobians.Jx
        Ju = nlp.constraint_jacobians.Ju
        # ∇²Lx .+= Jx' * (D * (Jx * z)) .+ Jx' * (D * (Ju * w))
        # ∇²Lu .+= Ju' * (D * (Jx * z)) .+ Ju' * (D * (Ju * w))
        mul!(diagjac, Jx, z)
        mul!(diagjac, Ju, w, 1.0, 1.0)
        diagjac .*= D
        mul!(∇²Lx, Jx', diagjac, 1.0, 1.0)
        mul!(∇²Lu, Ju', diagjac, 1.0, 1.0)
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

