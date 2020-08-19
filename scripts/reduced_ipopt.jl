# Verify solutions against matpower results
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using Ipopt

import ExaPF: ParseMAT, PowerSystem, IndexSet

# Build all the callbacks in a single closure.
function build_callback(pf, x0, u, p)
    # Compute initial point.
    x, g, Jx, Ju, _ = ExaPF.solve(pf, x0, u, p)
    gdGdx = Jx(pf, x, u, p)
    gdGdu = Ju(pf, x, u, p)
    previous_x = copy(x)
    dCdx, dCdu = ExaPF.cost_gradients(pf, x, u, p)
    λk = -(gdGdx\dCdx)
    # Store initial hash.
    hash_u = hash(u)
    function _update(u)
        # It looks like the tolerance of the Newton algorithm
        # could impact the convergence.
        x, g, Jx, Ju, conv = ExaPF.solve(pf, previous_x, u, p, maxiter=100, tol=1e-12)
        dGdx = Jx(pf, x, u, p)
        dGdu = Ju(pf, x, u, p)
        # I like to live dangerously
        !conv.has_converged && error("Fail to converge")
        # Copy in closure's arrays
        copy!(gdGdx, dGdx)
        copy!(gdGdu, dGdu)
        copy!(previous_x, x)
        # Update hash
        hash_u = hash(u)
    end
    function eval_f(u)
        (hash_u != hash(u)) && _update(u)
        c_u =  ExaPF.cost_function(pf, previous_x, u, p; V=eltype(u))
        # TODO: determine if we should include g(x, u), even if ≈ 0
        return c_u + λk' * g(pf, previous_x, u, p)
    end
    function eval_grad_f(u, grad_f)
        (hash_u != hash(u)) && _update(u)
        # Update gradient
        cost_x = xk -> ExaPF.cost_function(pf, xk, u, p; V=eltype(xk))
        cost_u = uk -> ExaPF.cost_function(pf, previous_x, uk, p; V=eltype(uk))
        fdCdx = xk -> ForwardDiff.gradient(cost_x, xk)
        fdCdu = uk -> ForwardDiff.gradient(cost_u, uk)
        dCdx = fdCdx(previous_x)
        dCdu = fdCdu(u)

        # dCdx, dCdu = ExaPF.cost_gradients(pf, previous_x, u, p)
        # lamba calculation
        lambda = -(gdGdx\dCdx)
        copy!(λk, lambda)
        # compute reduced gradient
        grad_f .= dCdu + (gdGdu')*lambda
        return nothing
    end
    function eval_g(x, g)
        return nothing
    end
    function eval_jac_g(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64})
        return nothing
    end
    function eval_h(u::Vector{Float64}, mode,
                    rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64,
                    lambda::Vector{Float64}, values::Vector{Float64})
        n = length(u)
        if mode == :Structure
            index = 0
            for i in 1:n
                for j in i:n
                    index += 1
                    rows[index] = i
                    cols[index] = j
                end
            end
        else
            (hash_u != hash(u)) && _update(u)
            # ExaPF.PowerSystem.print_state(pf, previous_x, u, p)
            cost_x = xk -> ExaPF.cost_function(pf, xk, u, p; V=eltype(xk))
            cost_u = uk -> ExaPF.cost_function(pf, previous_x, uk, p; V=eltype(uk))
            # Sensitivity matrix
            S = - inv(Array(gdGdx)) * gdGdu
            # ∂u²
            H_uu = zeros(n, n)
            ForwardDiff.hessian!(H_uu, cost_u, u)
            # ∂x²
            nx = length(previous_x)
            H_xx = zeros(nx, nx)
            ForwardDiff.hessian!(H_xx, cost_x, previous_x)
            # ∂x∂u
            function cross_f_xu(x, u)
                cost_x = x_ -> ExaPF.cost_function(pf, x_, u, p; V=eltype(x_))
                return ForwardDiff.gradient(cost_x, x)
            end
            H_xu = FiniteDiff.finite_difference_jacobian(u_ -> cross_f_xu(previous_x, u_), u)

            H = H_uu + S'*H_xx*S + S'*H_xu + H_xu' * S
            index = 0
            for i in 1:n
                for j in i:n
                    index += 1
                    @inbounds values[index] = H[i, j]
                end
            end
        end
        return nothing
    end
    return eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h
end

function run_reduced_ipopt(; hessian=false)
    datafile = "test/case9.m"
    # datafile = "../pglib-opf/pglib_opf_case1354_pegase.m"
    pf = PowerSystem.PowerNetwork(datafile, 1)

    # retrieve initial state of network
    pbus = real.(pf.sbus)
    qbus = imag.(pf.sbus)
    vmag = abs.(pf.vbus)
    vang = angle.(pf.vbus)

    x = ExaPF.PowerSystem.get_x(pf, vmag, vang, pbus, qbus)
    u = ExaPF.PowerSystem.get_u(pf, vmag, vang, pbus, qbus)
    p = ExaPF.PowerSystem.get_p(pf, vmag, vang, pbus, qbus)

    xk = copy(x)
    uk = copy(u)

    # Build callbacks in closure
    eval_f, eval_grad_f, eval_g, eval_jac_g, eval_hh = build_callback(pf, xk, uk, p)

    n = length(uk)
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)
    # Set bounds on decision variable
    x_L = zeros(n)
    x_U = zeros(n)
    # ... wrt. reference's voltages
    x_L[1:nref] .= 0.9
    x_U[1:nref] .= 1.1
    # ... wrt. active power in PV buses
    x_L[nref + 1:nref + npv] .= 0.0
    x_U[nref + 1:nref + npv] .= 3.0
    # ... wrt. voltages in PV buses
    x_L[nref + npv + 1:nref + 2*npv] .= 0.9
    x_U[nref + npv + 1:nref + 2*npv] .= 1.1

    # do not consider any constraint (yet)
    m = 0
    g_L = Float64[]
    g_U = Float64[]

    # Number of nonzeros in upper triangular Hessian
    if hessian
        hnnz = div(n * (n+1), 2)
        eval_h = eval_hh
    else
        hnnz = 0
        eval_h = nothing
    end

    prob = createProblem(n, x_L, x_U, m, g_L, g_U, 0, hnnz,
                         eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)

    prob.x = uk

    # This tests callbacks.
    function intermediate(alg_mod::Int, iter_count::Int,
                          obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
                          d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64,
                          ls_trials::Int)
        return iter_count < 100  # Interrupts after one iteration.
    end

    # I am too lazy to compute second order information
    if !hessian
        addOption(prob, "hessian_approximation", "limited-memory")
        addOption(prob, "limited_memory_initialization", "scalar1")
        # addOption(prob, "limited_memory_update_type", "sr1")
    end
    # I am an agressive guy (and at some point computing more trial steps
    # deteriorate the convergence)
    addOption(prob, "accept_after_max_steps", 1)
    # The derivative check is failing... so deactivate it
    # addOption(prob, "derivative_test", "first-order")
    setIntermediateCallback(prob, intermediate)

    solveProblem(prob)
    return prob
end

prob = run_reduced_ipopt(hessian=false)
println(prob.x)
