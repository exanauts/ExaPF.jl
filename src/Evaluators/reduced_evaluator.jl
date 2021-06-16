
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
mutable struct ReducedSpaceEvaluator{T, VI, VT, MT, Jacx, Jacu, JacCons, HessLag} <: AbstractNLPEvaluator
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
    obj_stack::AutoDiff.TapeMemory{typeof(cost_production), AdjointStackObjective{VT}, Nothing}
    cons_stacks::Vector{AutoDiff.TapeMemory} # / constraints
    constraint_jacobians::JacCons
    hesslag::HessLag

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
    powerflow_solver=NewtonRaphson(tol=1e-12),
    want_jacobian=true,
    nbatch_hessian=1,
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

    # Build Linear Algebra
    J = powerflow_jacobian(model)
    linear_solver = DirectSolver(J)

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
    want_hessian = (nbatch_hessian > 0)
    hess_ad = nothing
    if want_hessian
        hess_ad = if nbatch_hessian > 1
            BatchHessianLagrangian(model, J, nbatch_hessian)
        else
            HessianLagrangian(model, J)
        end
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
has_hessian(nlp::ReducedSpaceEvaluator) = nlp.has_hessian
number_batches_hessian(nlp::ReducedSpaceEvaluator) = nlp.has_hessian ? n_batches(nlp.hesslag) : 0

# Getters
get(nlp::ReducedSpaceEvaluator, ::Constraints) = nlp.constraints
function get(nlp::ReducedSpaceEvaluator, ::State)
    x = similar(nlp.λ) ; fill!(x, 0)
    get!(nlp.model, State(), x, nlp.buffer)
    return x
end
get(nlp::ReducedSpaceEvaluator, ::PhysicalState) = nlp.buffer

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

    # Evaluate Jacobian of power flow equation on current u
    AutoDiff.jacobian!(nlp.model, nlp.state_jacobian.u, nlp.buffer)
    # Specify that constraint's Jacobian is not up to date
    nlp.update_jacobian = nlp.has_jacobian
    # Update Hessian factorization
    if !isnothing(nlp.hesslag)
        ∇gₓ = nlp.state_jacobian.x.J
        update_factorization!(nlp.hesslag, ∇gₓ)
        # Update values for Hessian's AutoDiff
        update_hessian!(nlp.model, nlp.hesslag.hess, nlp.buffer)
    end
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

function _backward_solve!(nlp::ReducedSpaceEvaluator, y, x)
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

function jprod!(nlp::ReducedSpaceEvaluator, jm, u, v)
    nᵤ = length(u)
    m  = n_constraints(nlp)
    @assert nᵤ == size(v, 1)

    _update_full_jacobian_constraints!(nlp)
    H = nlp.hesslag
    ∇gᵤ = nlp.state_jacobian.u.J

    # Arrays
    Jx = nlp.constraint_jacobians.Jx
    Ju = nlp.constraint_jacobians.Ju
    z = H.z

    # init RHS
    mul!(z, ∇gᵤ, v)
    LinearAlgebra.ldiv!(H.lu, z)

    # jv .= Ju * v .- Jx * z
    mul!(jm, Ju, v)
    mul!(jm, Jx, z, -1.0, 1.0)
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
    @assert !isnothing(nlp.hesslag)
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
# Single version
function full_hessprod!(nlp::ReducedSpaceEvaluator, hv::AbstractVector, y::AbstractVector, tgt::AbstractVector)
    nx, nu = get(nlp.model, NumberOfState()), get(nlp.model, NumberOfControl())
    H = nlp.hesslag
    AutoDiff.adj_hessian_prod!(nlp.model, H.hess, hv, nlp.buffer, y, tgt)
    ∂fₓ = @view hv[1:nx]
    ∂fᵤ = @view hv[nx+1:nx+nu]
    return ∂fₓ , ∂fᵤ
end

# Batch version
function full_hessprod!(nlp::ReducedSpaceEvaluator, hv::AbstractMatrix, y::AbstractMatrix, tgt::AbstractMatrix)
    nx, nu = get(nlp.model, NumberOfState()), get(nlp.model, NumberOfControl())
    H = nlp.hesslag
    batch_adj_hessian_prod!(nlp.model, H.hess, hv, nlp.buffer, y, tgt)
    ∂fₓ = hv[1:nx, :]
    ∂fᵤ = hv[nx+1:nx+nu, :]
    return ∂fₓ , ∂fᵤ
end

function hessprod!(nlp::ReducedSpaceEvaluator, hessvec, u, w)
    @assert nlp.hesslag != nothing

    nx = get(nlp.model, NumberOfState())
    nu = get(nlp.model, NumberOfControl())
    H = nlp.hesslag
    ∇gᵤ = nlp.state_jacobian.u.J

    # Number of batches
    nbatch = size(w, 2)
    @assert nbatch == size(H.z, 2) == size(hessvec, 2)

    # Load variables and buffers
    tgt = H.tmp_tgt
    hv = H.tmp_hv
    y = H.y
    z = H.z
    ψ = H.ψ

    # Step 1: computation of first second-order adjoint
    mul!(z, ∇gᵤ, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(H.lu, z)

    # Init tangent with z and w
    for i in 1:nbatch
        mxu = 1 + (i-1)*(nx+nu)
        mx = 1 + (i-1)*nx
        mu = 1 + (i-1)*nu
        copyto!(tgt, mxu,    z, mx, nx)
        copyto!(tgt, mxu+nx, w, mu, nu)
    end

    # Init adjoint
    fill!(y, 0.0)
    y[end] = 1.0       # / objective
    y[1:nx] .-= nlp.λ  # / power balance

    # STEP 2: AutoDiff
    ∂fₓ, ∂fᵤ = full_hessprod!(nlp, hv, y, tgt)

    # STEP 3: computation of second second-order adjoint
    copyto!(ψ, ∂fₓ)
    LinearAlgebra.ldiv!(H.adjlu, ψ)

    hessvec .= ∂fᵤ
    mul!(hessvec, transpose(∇gᵤ), ψ, -1.0, 1.0)

    return
end

function hessian_lagrangian_penalty_prod!(
    nlp::ReducedSpaceEvaluator, hessvec, u, y, σ, D, w,
)
    @assert nlp.hesslag != nothing

    nbatch = size(w, 2)
    nx = get(nlp.model, NumberOfState())
    nu = get(nlp.model, NumberOfControl())
    buffer = nlp.buffer
    H = nlp.hesslag
    ∇gᵤ = nlp.state_jacobian.u.J

    fill!(hessvec, 0.0)

    z = H.z
    ψ = H.ψ
    ∇gᵤ = nlp.state_jacobian.u.J
    mul!(z, ∇gᵤ, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(H.lu, z)

    # Two vector products
    μ = H.y
    tgt = H.tmp_tgt
    hv = H.tmp_hv

    # Init tangent with z and w
    for i in 1:nbatch
        mxu = 1 + (i-1)*(nx+nu)
        mx = 1 + (i-1)*nx
        mu = 1 + (i-1)*nu
        copyto!(tgt, mxu,    z, mx, nx)
        copyto!(tgt, mxu+nx, w, mu, nu)
    end

    ## OBJECTIVE HESSIAN
    fill!(μ, 0.0)
    μ[1:nx] .-= nlp.λ  # / power balance
    μ[end] = σ         # / objective
    # / constraints
    shift_m = nx
    shift_y = size_constraint(nlp.model, voltage_magnitude_constraints)
    for cons in nlp.constraints
        isa(cons, typeof(voltage_magnitude_constraints)) && continue
        m = size_constraint(nlp.model, cons)::Int
        μ[shift_m+1:m+shift_m] .= view(y, shift_y+1:shift_y+m)
        shift_m += m
        shift_y += m
    end

    ∇²Lx, ∇²Lu = full_hessprod!(nlp, hv, μ, tgt)

    # Add Hessian of quadratic penalty
    m = length(y)
    diagjac = (nbatch > 1) ? similar(y, m, nbatch) : similar(y)
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

    # Second order adjoint
    copyto!(ψ, ∇²Lx)
    LinearAlgebra.ldiv!(H.adjlu, ψ)

    hessvec .+= ∇²Lu
    mul!(hessvec, transpose(∇gᵤ), ψ, -1.0, 1.0)

    return
end

# Batch Hessian
macro define_batch_hessian(function_name, target_function, args...)
    fname = Symbol(function_name)
    argstup = Tuple(args)
    quote
        function $(esc(fname))(nlp::ReducedSpaceEvaluator, dest, $(map(esc, argstup)...))
            @assert has_hessian(nlp)
            n = ExaPF.n_variables(nlp)
            ∇²f = nlp.hesslag.hess
            nbatch = size(nlp.hesslag.tmp_hv, 2)

            # Allocate memory
            v_cpu = zeros(n, nbatch)
            v = similar(x, n, nbatch)

            N = div(n, nbatch, RoundDown)
            for i in 1:N
                # Init tangents on CPU
                fill!(v_cpu, 0.0)
                @inbounds for j in 1:nbatch
                    v_cpu[j+(i-1)*nbatch, j] = 1.0
                end
                # Pass tangents to the device
                copyto!(v, v_cpu)

                hm = @view dest[:, nbatch * (i-1) + 1: nbatch * i]
                $target_function(nlp, hm, $(map(esc, argstup)...), v)
            end

            # Last slice
            last_batch = n - N*nbatch
            if last_batch > 0
                fill!(v_cpu, 0.0)
                @inbounds for j in 1:nbatch
                    v_cpu[n-nbatch+j, j] = 1.0
                end
                copyto!(v, v_cpu)

                hm = @view dest[:, (n - nbatch + 1) : n]
                $target_function(nlp, hm, $(map(esc, argstup)...), v)
            end
        end
    end
end

@define_batch_hessian hessian! hessprod! x
@define_batch_hessian hessian_lagrangian_penalty! hessian_lagrangian_penalty_prod! x y σ D
@define_batch_hessian jacobian! jprod! x


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
    print(io, "    * linear solver: ", typeof(nlp.linear_solver))
end

function reset!(nlp::ReducedSpaceEvaluator)
    # Reset adjoint
    fill!(nlp.λ, 0)
    # Reset buffer
    init_buffer!(nlp.model, nlp.buffer)
    return
end

