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
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)
    # Compute initial point.
    xk, g, Jx, Ju, _ = ExaPF.solve(pf, x0, u, p)
    ∇gₓ = Jx(pf, xk, u, p)
    ∇gᵤ = Ju(pf, xk, u, p)
    ∇fₓ, ∇fᵤ = ExaPF.cost_gradients(pf, xk, u, p)
    λk = -(∇gₓ'\∇fₓ)
    # Store initial hash.
    hash_u = hash(u)
    function _update(u)
        # It looks like the tolerance of the Newton algorithm
        # could impact the convergence.
        x, g, Jx, Ju, conv = ExaPF.solve(pf, xk, u, p, maxiter=20, tol=1e-10)
        dGdx = Jx(pf, x, u, p)
        dGdu = Ju(pf, x, u, p)
        # I like to live dangerously
        if !conv.has_converged
            println(u)
            println(conv.norm_residuals)
            error("Fail to converge")
        end
        # Copy in closure's arrays
        copy!(∇gₓ, dGdx)
        copy!(∇gᵤ, dGdu)
        copy!(xk, x)
        # Update hash
        hash_u = hash(u)
    end
    function eval_f(u)
        (hash_u != hash(u)) && _update(u)
        c_u =  ExaPF.cost_function(pf, xk, u, p; V=eltype(u))
        # TODO: determine if we should include g(x, u), even if ≈ 0
        return c_u + λk' * g(pf, xk, u, p)
    end
    function eval_grad_f(u, grad_f)
        (hash_u != hash(u)) && _update(u)
        # Update gradient
        cost_x = x_ -> ExaPF.cost_function(pf, x_, u, p; V=eltype(x_))
        cost_u = u_ -> ExaPF.cost_function(pf, xk, u_, p; V=eltype(u_))
        fdCdx = xk -> ForwardDiff.gradient(cost_x, xk)
        fdCdu = uk -> ForwardDiff.gradient(cost_u, uk)
        ∇fₓ = fdCdx(xk)
        ∇fᵤ = fdCdu(u)

        # S = - inv(Array(∇gₓ))' * ∇gᵤ
        # grad_f .= ∇fᵤ + S' * ∇fₓ
        # dCdx, dCdu = ExaPF.cost_gradients(pf, previous_x, u, p)
        # lamba calculation
        λk = -(∇gₓ'\∇fₓ)
        # compute reduced gradient
        grad_f .= ∇fᵤ + (∇gᵤ')*λk
        return nothing
    end
    function eval_g(u, g)
        (hash_u != hash(u)) && _update(u)
        g .= xk[1:npq]
        return nothing
    end
    function eval_jac_g(u::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64})
        n = length(u)
        if mode == :Structure
            idx = 1
            for c in 1:npq #number of constraints
                for i in 1:n # number of variables
                    rows[idx] = c ; cols[idx] = i
                    idx += 1
                end
            end
        else
            (hash_u != hash(u)) && _update(u)
            nx = length(xk)
            n = length(u)
            J = zeros(npq, n)
            rhs = zeros(nx)
            λ = zeros(nx)
            for ix in 1:npq
                rhs .= 0.0
                rhs[ix] = 1.0
                λ .= - ∇gₓ' \ rhs
                J[ix, :] .= ∇gᵤ' * λ
            end
            k = 1
            for i in 1:npq
                for j in 1:n
                    values[k] = J[i, j]
                    k += 1
                end
            end
        end
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
            cost_x = x_ -> ExaPF.cost_function(pf, x_, u, p; V=eltype(x_))
            cost_u = u_ -> ExaPF.cost_function(pf, xk, u_, p; V=eltype(u_))
            # Sensitivity matrix
            S = - inv(Array(∇gₓ)) * gdGdu
            # ∂u²
            H_uu = zeros(n, n)
            ForwardDiff.hessian!(H_uu, cost_u, u)
            # ∂x²
            nx = length(xk)
            H_xx = zeros(nx, nx)
            ForwardDiff.hessian!(H_xx, cost_x, xk)
            # ∂x∂u
            function cross_f_xu(x, u)
                cost_x = x_ -> ExaPF.cost_function(pf, x_, u, p; V=eltype(x_))
                return ForwardDiff.gradient(cost_x, x)
            end
            H_xu = FiniteDiff.finite_difference_jacobian(u_ -> cross_f_xu(xk, u_), u)

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

function run_reduced_ipopt(; hessian=false, cons=false)
    datafile = "test/case9.m"
    # datafile = "test/case14.raw"
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
    n = length(uk)
    uk .= max.(uk, 0.0)

    # Build callbacks in closure
    eval_f, eval_grad_f, eval_g, eval_jac_g, eval_hh = build_callback(pf, xk, uk, p)

    v_min = 0.9
    v_max = 1.1
    u_min, u_max, x_min, x_max = ExaPF.get_constraints(pf)

    # Set bounds on decision variable
    x_L = u_min
    x_U = u_max

    # add constraint on PQ's voltage magnitude
    if cons
        m = npq
        jnnz = m * n
        g_L = fill(v_min, m)
        g_U = fill(v_max, m)
    else
        m = 0
        jnnz = 0
        g_L = Float64[]
        g_U = Float64[]
    end

    # Number of nonzeros in upper triangular Hessian
    if hessian
        hnnz = div(n * (n+1), 2)
        eval_h = eval_hh
    else
        hnnz = 0
        eval_h = nothing
    end

    prob = Ipopt.createProblem(n, x_L, x_U, m, g_L, g_U, jnnz, hnnz,
                         eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)

    # prob.x = (u_min .+ u_max) ./ 2.0
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
    addOption(prob, "accept_after_max_steps", 10)
    # The derivative check is failing... so deactivate it
    addOption(prob, "derivative_test", "first-order")
    Ipopt.setIntermediateCallback(prob, intermediate)

    # println(xk)
    Ipopt.solveProblem(prob)
    u = prob.x
    x, g, Jx, Ju, conv = ExaPF.solve(pf, xk, u, p, maxiter=50, tol=1e-14)
    ExaPF.PowerSystem.print_state(pf, x, u, p)
    return prob
end

prob = run_reduced_ipopt(hessian=false, cons=true)
println(prob.x)
