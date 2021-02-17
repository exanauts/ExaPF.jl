
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
mutable struct ReducedSpaceEvaluator{T, VI, VT, MT} <: AbstractNLPEvaluator
    model::PolarForm{T, VI, VT, MT}
    λ::VT

    u_min::VT
    u_max::VT

    constraints::Array{Function, 1}
    g_min::VT
    g_max::VT

    buffer::PolarNetworkState{VI, VT}
    autodiff::AutoDiffFactory{VT}
    ∇gᵗ::AbstractMatrix
    linear_solver::LinearSolvers.AbstractLinearSolver
    powerflow_solver::AbstractNonLinearSolver
end

function ReducedSpaceEvaluator(
    model, x, u;
    constraints=Function[state_constraints, power_constraints],
    linear_solver=DirectSolver(),
    powerflow_solver=NewtonRaphson(tol=1e-12),
)
    # First, build up a network buffer
    buffer = get(model, PhysicalState())
    # Populate buffer with default values of the network, as stored
    # inside model
    init_buffer!(model, buffer)
    # Build up AutoDiff factory
    jx, ju, adjoint_f = init_autodiff_factory(model, buffer)
    ad = AutoDiffFactory(jx, ju, adjoint_f)

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

    return ReducedSpaceEvaluator(
        model, λ, u_min, u_max,
        constraints, g_min, g_max,
        buffer,
        ad, jx.J, linear_solver, powerflow_solver,
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
    return get(nlp.model, Control(), nlp.buffer)
end

# Bounds
bounds(nlp::ReducedSpaceEvaluator, ::Variables) = (nlp.u_min, nlp.u_max)
bounds(nlp::ReducedSpaceEvaluator, ::Constraints) = (nlp.g_min, nlp.g_max)

function update!(nlp::ReducedSpaceEvaluator, u)
    jac_x = nlp.autodiff.Jgₓ
    # Transfer control u into the network cache
    transfer!(nlp.model, nlp.buffer, u)
    # Get corresponding point on the manifold
    conv = powerflow(nlp.model, jac_x, nlp.buffer, nlp.powerflow_solver;
                     solver=nlp.linear_solver)
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

    # Refresh values of active and reactive powers at generators
    update!(nlp.model, PS.Generators(), PS.ActivePower(), nlp.buffer)
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
    jacobian(nlp.model, nlp.autodiff.Jgᵤ, nlp.buffer, AutoDiff.ControlJacobian())
end

# compute inplace reduced gradient (g = ∇fᵤ + (∇gᵤ')*λ)
# equivalent to: g = ∇fᵤ - (∇gᵤ')*λ_neg
# (take λₖ_neg to avoid computing an intermediate array)
function reduced_gradient!(
    nlp::ReducedSpaceEvaluator, grad, ∂fₓ, ∂fᵤ, λ, u,
)
    ∇gᵤ = nlp.autodiff.Jgᵤ.J
    # Compute adjoint and store value inside λₖ
    LinearSolvers.ldiv!(nlp.linear_solver, λ, nlp.∇gᵗ, ∂fₓ)
    grad .= ∂fᵤ
    mul!(grad, transpose(∇gᵤ), λ, -1.0, 1.0)
end
function reduced_gradient!(nlp::ReducedSpaceEvaluator, grad, ∂fₓ, ∂fᵤ, u)
    reduced_gradient!(nlp::ReducedSpaceEvaluator, grad, ∂fₓ, ∂fᵤ, nlp.λ, u)
end

# Compute only full gradient wrt x and u
function full_gradient!(nlp::ReducedSpaceEvaluator, gx, gu, u)
    buffer = nlp.buffer
    ∂obj = nlp.autodiff.∇f
    # Evaluate adjoint of cost function and update inplace AdjointStackObjective
    ∂cost(nlp.model, ∂obj, buffer)
    copyto!(gx, ∂obj.∇fₓ)
    copyto!(gu, ∂obj.∇fᵤ)
end

function gradient!(nlp::ReducedSpaceEvaluator, g, u)
    buffer = nlp.buffer
    # Evaluate adjoint of cost function and update inplace AdjointStackObjective
    ∂cost(nlp.model, nlp.autodiff.∇f, buffer)
    ∇fₓ, ∇fᵤ = nlp.autodiff.∇f.∇fₓ, nlp.autodiff.∇f.∇fᵤ

    # Evaluate Jacobian of power flow equation on current u
    update_jacobian!(nlp, Control())
    reduced_gradient!(nlp, g, ∇fₓ, ∇fᵤ, u)
    return
end

function constraint!(nlp::ReducedSpaceEvaluator, g, u)
    ϕ = nlp.buffer
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

function full_jacobian(nlp::ReducedSpaceEvaluator, u)
    jacobians_x = SparseMatrixCSC{Float64, Int}[]
    jacobians_u = SparseMatrixCSC{Float64, Int}[]
    for cons in nlp.constraints
        J = jacobian(nlp.model, cons, nlp.buffer)::FullSpaceJacobian{SparseMatrixCSC{Float64, Int}}
        push!(jacobians_x, J.x)
        push!(jacobians_u, J.u)
    end
    # TODO: too many allocations there
    Jx = vcat(jacobians_x...)
    Ju = vcat(jacobians_u...)

    return FullSpaceJacobian(Jx, Ju)
end

function jacobian!(nlp::ReducedSpaceEvaluator, jac, u)
    J = full_jacobian(nlp, u)
    m, nₓ = size(J.x)
    ∇gᵤ = nlp.autodiff.Jgᵤ.J
    # Compute factorization with UMFPACK
    ∇gfac = factorize(nlp.∇gᵗ)
    # Compute adjoints all in once, using the same factorization
    μ = zeros(nₓ, m)
    ldiv!(μ, ∇gfac, J.x')
    # Compute reduced Jacobian
    copy!(jac, J.u)
    mul!(jac, μ', ∇gᵤ, -1.0, 1.0)
    return
end

function full_jtprod!(nlp::ReducedSpaceEvaluator, jvx, jvu, u, v)
    ∂obj = nlp.autodiff.∇f
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
end

function jtprod!(nlp::ReducedSpaceEvaluator, jv, u, v)
    ∂obj = nlp.autodiff.∇f
    jvx = ∂obj.jvₓ ; fill!(jvx, 0)
    jvu = ∂obj.jvᵤ ; fill!(jvu, 0)
    full_jtprod!(nlp, jvx, jvu, u, v)
    reduced_gradient!(nlp, jv, jvx, jvu, jvx, u)
    return
end

function jtprod!(nlp::ReducedSpaceEvaluator, cons::Function, jv, u, v)
    model = nlp.model
    ∂obj = nlp.autodiff.∇f
    # Get adjoint
    jtprod(model, cons, ∂obj, nlp.buffer, v)
    jvx, jvu = ∂obj.∇fₓ, ∂obj.∇fᵤ
    reduced_gradient!(nlp, jv, jvx, jvx, jvu)
    return
end

function ojtprod!(nlp::ReducedSpaceEvaluator, jv, u, σ, v)
    ∂obj = nlp.autodiff.∇f
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
    return
end

# Compute reduced Hessian-vector product using the adjoint-adjoint method
function reduced_hessian!(
    nlp::ReducedSpaceEvaluator, hessvec, ∇²fₓₓ::SpMT, ∇²fₓᵤ::SpMT, ∇²fᵤᵤ::SpMT, w,
) where SpMT
    λ = -nlp.λ # take care that λ is negative of true adjoint!
    # Jacobian
    ∇gₓ = nlp.autodiff.Jgₓ.J::SpMT
    ∇gᵤ = nlp.autodiff.Jgᵤ.J::SpMT

    # Evaluate Hess-vec of residual function g(x, u) = 0
    ∇²gλ = residual_hessian(nlp.model, nlp.buffer, λ)::FullSpaceHessian{SpMT}

    # Adjoint-adjoint
    ∇gaₓ = ∇²fₓₓ + ∇²gλ.xx
    z = -(∇gₓ ) \ (∇gᵤ * w)
    ψ = -(∇gₓ') \ (∇²fₓᵤ' * w + ∇²gλ.xu' * w +  ∇gaₓ * z)
    hessvec .= ∇²fᵤᵤ * w +  ∇²gλ.uu * w + ∇gᵤ' * ψ  + ∇²fₓᵤ * z + ∇²gλ.xu * z
    return
end

function hessprod!(nlp::ReducedSpaceEvaluator, hessvec, u, w)
    buffer = nlp.buffer
    ∂f = nlp.autodiff.∇f
    ∇²f = hessian_cost(nlp.model, ∂f, buffer)::FullSpaceHessian{SparseMatrixCSC{Float64, Int}}
    reduced_hessian!(nlp, hessvec, ∇²f.xx, ∇²f.xu, ∇²f.uu, w)
    return
end

function full_hessian_lagrangian(nlp::ReducedSpaceEvaluator, u, y, σ)
    nu = n_variables(nlp)
    buffer = nlp.buffer
    ∂f = nlp.autodiff.∇f
    ∇²f = hessian_cost(nlp.model, ∂f, buffer)::FullSpaceHessian{SparseMatrixCSC{Float64, Int}}
    ∇²Lₓₓ = σ .* ∇²f.xx
    ∇²Lₓᵤ = σ .* ∇²f.xu
    ∇²Lᵤᵤ = σ .* ∇²f.uu

    shift = 0
    for cons in nlp.constraints
        m = size_constraint(nlp.model, cons)
        mask = shift+1:shift+m
        yc = @view y[mask]
        ∇²h = hessian(nlp.model, cons, ∂f, nlp.buffer, yc)::FullSpaceHessian{SparseMatrixCSC{Float64, Int}}
        ∇²Lₓₓ .+= ∇²h.xx
        ∇²Lₓᵤ .+= ∇²h.xu
        ∇²Lᵤᵤ .+= ∇²h.uu
        shift += m
    end

    return FullSpaceHessian(∇²Lₓₓ, ∇²Lₓᵤ, ∇²Lᵤᵤ)
end

function hessian_lagrangian_prod!(
    nlp::ReducedSpaceEvaluator, hessvec, u, y, σ, v,
)
    # Full Hessian of Lagrangian L(u, y) = f(u) + y' * h(u)
    ∇²L = full_hessian_lagrangian(nlp, u, y, σ)::FullSpaceHessian{SparseMatrixCSC{Float64, Int}}
    reduced_hessian!(nlp, hessvec, ∇²L.xx, ∇²L.xu, ∇²L.uu, v)
    return
end

function hessian_lagrangian_penalty_prod!(
    nlp::ReducedSpaceEvaluator, hessvec, u, y, σ, v, w,
)
    # Full Hessian of Lagrangian L(u, y) = f(u) + y' * h(u)
    ∇²L = full_hessian_lagrangian(nlp, u, y, σ)::FullSpaceHessian{SparseMatrixCSC{Float64, Int}}
    # Add Hessian of quadratic penalty
    J = full_jacobian(nlp, u)::FullSpaceJacobian{SparseMatrixCSC{Float64, Int}}
    D = Diagonal(w)
    ∇²L.xx .+= J.x' * D * J.x
    ∇²L.xu .+= J.u' * D * J.x
    ∇²L.uu .+= J.u' * D * J.u
    reduced_hessian!(nlp, hessvec, ∇²L.xx, ∇²L.xu, ∇²L.uu, v)
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
end

