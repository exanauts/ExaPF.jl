# Verify solutions against matpower results
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using SparseArrays
using Ipopt

import ExaPF: ParseMAT, PowerSystem, IndexSet

get_lines_limit(pf::PowerSystem.PowerNetwork) = pf.data["branch"][:, 6]

function flow_limit(pf, x, u, p; T=Float64)
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)
    b2i = pf.bus_to_indexes
    branches = pf.data["branch"]
    nlines = size(branches, 1)
    cons_fr = zeros(T, nlines)
    cons_to = zeros(T, nlines)
    Vm, Va, pbus, qbus = ExaPF.PowerSystem.retrieve_physics(pf, x, u, p; V=V)

    for i in 1:nlines
        bus_fr = Int(branches[i, 1])
        cons_fr[i] = pbus[bus_fr]^2 + qbus[bus_fr]^2
        bus_to = Int(branches[i, 2])
        cons_to[i] = pbus[bus_to]^2 + qbus[bus_to]^2
    end

    return [cons_from; cons_to]
end

function gₚ(pf, x, u, p; V=Float64)
    nref = length(pf.ref)
    npv = length(pf.pv)
    ybus_re, ybus_im = ExaPF.Spmat{Vector}(pf.Ybus)

    Vm, Va, pbus, qbus = PowerSystem.retrieve_physics(pf, x, u, p; V=V)
    p_ref = zeros(V, nref)
    q_pv = zeros(V, npv)

    for (i, bus) in enumerate(pf.ref)
        p_ref[i] = PowerSystem.get_power_injection(bus, Vm, Va, ybus_re, ybus_im)
    end
    for (i, bus) in enumerate(pf.pv)
        q_pv[i] = PowerSystem.get_react_injection(bus, Vm, Va, ybus_re, ybus_im)
    end
    return [p_ref; q_pv]
end

# Build all the callbacks in a single closure.
function build_callback(pf, x0, u, p)
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)
    # Compute initial point.
    xk, g, Jx, Ju, conv, resx! = ExaPF.solve(pf, x0, u, p, tol=1e-13)
    ∇gₓ = Jx(pf, xk, u, p)
    ∇gᵤ = Ju(pf, xk, u, p)
    ∇fₓ, ∇fᵤ = ExaPF.cost_gradients(pf, xk, u, p)
    λk = -(∇gₓ'\∇fₓ)
    # Store initial hash.
    hash_u = hash(u)
    function _update(u)
        # It looks like the tolerance of the Newton algorithm
        # could impact the convergence.
        x, g, Jx, Ju, conv, resx! = ExaPF.solve(pf, xk, u, p, maxiter=50, tol=5e-12)
        dGdx = Jx(pf, x, u, p)
        dGdu = Ju(pf, x, u, p)
        # I like to live dangerously
        if !conv.has_converged
            println(u)
            println(xk)
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

        # lamba calculation
        λk = -(∇gₓ'\∇fₓ)
        # compute reduced gradient
        grad_f .= ∇fᵤ + (∇gᵤ')*λk
        return nothing
    end
    function eval_g(u, g)
        (hash_u != hash(u)) && _update(u)
        # Constraints on vmag_{pq}
        g[1:npq] .= xk[1:npq]
        # Constraint on p_{ref}
        g[npq+1:npq+nref+npv] .= gₚ(pf, xk, u, p)
        return nothing
    end
    function eval_jac_g(u::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64})
        n = length(u)
        m = npq + nref + npv
        if mode == :Structure
            idx = 1
            for c in 1:m #number of constraints
                for i in 1:n # number of variables
                    rows[idx] = c ; cols[idx] = i
                    idx += 1
                end
            end
        else
            (hash_u != hash(u)) && _update(u)
            nx = length(xk)
            n = length(u)
            J = zeros(m, n)
            rhs = zeros(nx)
            λ = zeros(nx)
            # Evaluate reduced Jacobian for bounds on vmag_{pq}
            for ix in 1:npq
                rhs .= 0.0
                rhs[ix] = 1.0
                λ .= - ∇gₓ' \ rhs
                J[ix, :] .= ∇gᵤ' * λ
            end
            # Evaluate reduced Jacobian for bounds on p_{ref}
            gg_x(x_) = gₚ(pf, x_, u, p; V=eltype(x_))
            gg_u(u_) = gₚ(pf, xk, u_, p; V=eltype(u_))
            jac_p_x = ForwardDiff.jacobian(gg_x, xk)
            jac_p_u = ForwardDiff.jacobian(gg_u, u)
            for ix in 1:(nref + npv)
                rhs = jac_p_x[ix, :]
                λ .= - ∇gₓ' \ rhs
                J[npq + ix, :] .= jac_p_u[ix, :] + ∇gᵤ' * λ
            end

            # Copy to Ipopt's Jacobian
            k = 1
            for i in 1:m
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
            ## Hessian of cost function
            cost_x = x_ -> ExaPF.cost_function(pf, x_, u, p; V=eltype(x_))
            cost_u = u_ -> ExaPF.cost_function(pf, xk, u_, p; V=eltype(u_))
            # Sensitivity matrix
            S = - inv(Array(∇gₓ)) * ∇gᵤ
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
            #TODO
            H_xu = FiniteDiff.finite_difference_jacobian(u_ -> cross_f_xu(xk, u_), u)

            # Hessian of the equality constraint
            # Evaluate constraints
            vecx = [xk; u]
            fjac = vecx -> ForwardDiff.jacobian(resx!, vecx)
            jac = fjac(vecx)
            jacx = sparse(jac[:,1:nx])
            jacu = sparse(jac[:,nx+1:end])
            hes = ForwardDiff.jacobian(fjac, vecx)
            # I am not sure about the reshape.
            # It could be that length(xk) goes at the end. This tensor stuff is a brain twister.
            hes = reshape(hes, (nx, nx + n, nx + n))
            ghesxx = hes[:, 1:nx, 1:nx]
            ghesxu = hes[:, 1:nx, nx+1:end]
            ghesuu = hes[:, nx+1:end, nx+1:end]
            for i in 1:nx
                H_uu .+= λk[i] * ghesuu[i, :, :]
                H_xx .+= λk[i] * ghesxx[i, :, :]
                H_xu .+= λk[i] * ghesxu[i, :, :]
            end

            # Global Hessian
            H = H_uu + S'*H_xx*S + S'*H_xu + H_xu' * S
            H = obj_factor * (H + H') / 2.0

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

function run_reduced_ipopt(datafile; hessian=false, cons=false)
    pf = PowerSystem.PowerNetwork(datafile, 1)
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

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

    # Build callbacks in closure
    eval_f, eval_grad_f, eval_gg, eval_jac_gg, eval_hh = build_callback(pf, xk, uk, p)

    v_min = 0.9
    v_max = 1.1
    u_min, u_max, x_min, x_max, p_min, p_max = ExaPF.get_bound_constraints(pf)
    q_min, q_max = ExaPF.get_bound_reactive_power(pf)

    # Set bounds on decision variable
    x_L = u_min
    x_U = u_max

    # add constraint on PQ's voltage magnitude
    if cons
        m = npq + nref + npv
        jnnz = m * n
        g_L = [x_min[1:npq]; p_min; q_min]
        g_U = [x_max[1:npq]; p_max; q_max]
        eval_g = eval_gg
        eval_jac_g = eval_jac_gg
    else
        m = 0
        jnnz = 0
        g_L = Float64[]
        g_U = Float64[]
        eval_g(u, g) = nothing
        eval_jac_g(u::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64}) = nothing
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
        return iter_count < 500  # Interrupts after one iteration.
    end

    # I am too lazy to compute second order information
    if !hessian
        addOption(prob, "hessian_approximation", "limited-memory")
        addOption(prob, "limited_memory_initialization", "scalar1")
        addOption(prob, "limited_memory_max_history", 50)
        # addOption(prob, "limited_memory_update_type", "sr1")
    end
    # addOption(prob, "accept_after_max_steps", 10)
    addOption(prob, "tol", 1e-2)
    # addOption(prob, "derivative_test", "first-order")
    Ipopt.setIntermediateCallback(prob, intermediate)

    Ipopt.solveProblem(prob)
    u = prob.x
    x, g, Jx, Ju, conv = ExaPF.solve(pf, xk, u, p, maxiter=50, tol=1e-14)
    # ExaPF.PowerSystem.print_state(pf, x, u, p)
    return prob, pf
end

# datafile = "test/data/case9.m"
# datafile = "../pglib-opf/pglib_opf_case30_ieee.m"
# datafile = "../pglib-opf/pglib_opf_case57_ieee.m"
datafile = "../pglib-opf/pglib_opf_case118_ieee.m"
# datafile = "../pglib-opf/pglib_opf_case300_ieee.m"
# datafile = "../pglib-opf/pglib_opf_case1354_pegase.m"
# datafile = "../pglib-opf/pglib_opf_case1888_rte.m"
# datafile = "../pglib-opf/pglib_opf_case200_activ.m"
# datafile = "../pglib-opf/pglib_opf_case500_goc.m"
prob, pf = run_reduced_ipopt(datafile; hessian=false, cons=true)
prob.status

