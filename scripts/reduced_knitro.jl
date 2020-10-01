# Verify solutions against matpower results
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using KNITRO

import ExaPF: ParseMAT, PowerSystem, IndexSet

# Build all the callbacks in a single closure.
function build_callback(pf, x0, u, p)
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)
    # Compute initial point.
    xk, g, Jx, Ju, _ = ExaPF.solve(pf, x0, u, p, tol=1e-10)
    ∇gₓ = Jx(pf, xk, u, p)
    ∇gᵤ = Ju(pf, xk, u, p)
    ∇fₓ, ∇fᵤ = ExaPF.cost_gradients(pf, xk, u, p)
    λk = -(∇gₓ\∇fₓ)
    # Store initial hash.
    hash_u = hash(u)
    function _update(u)
        # It looks like the tolerance of the Newton algorithm
        # could impact the convergence.
        x, g, Jx, Ju, conv = ExaPF.solve(pf, xk, u, p, maxiter=50, tol=1e-14)
        dGdx = Jx(pf, x, u, p)
        dGdu = Ju(pf, x, u, p)
        # I like to live dangerously
        if !conv.has_converged
            @info("uₖ = ", u)
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
        return c_u #+ λk' * g(pf, xk, u, p)
    end
    function eval_grad_f(u, grad_f)
        (hash_u != hash(u)) && _update(u)
        # Update gradient
        cost_x = x_ -> ExaPF.cost_function(pf, x_, u, p; V=eltype(x_))
        cost_u = u_ -> ExaPF.cost_function(pf, xk, u_, p; V=eltype(u_))
        fdCdx = x_ -> ForwardDiff.gradient(cost_x, x_)
        fdCdu = u_ -> ForwardDiff.gradient(cost_u, u_)
        ∇fₓ = fdCdx(xk)
        ∇fᵤ = fdCdu(u)

        # S = - inv(Array(∇gₓ))' * ∇gᵤ
        # grad_f .= ∇fᵤ + S' * ∇fₓ
        # dCdx, dCdu = ExaPF.cost_gradients(pf, previous_x, u, p)
        # lamba calculation
        λk = -(∇gₓ')\∇fₓ
        grad_f .= ∇fᵤ + (∇gᵤ')*λk
        return nothing
    end
    function eval_g(u, g)
        (hash_u != hash(u)) && _update(u)
        g .= xk[1:npq]
        return nothing
    end
    function eval_jac_g(
        jac_g::Vector{Float64},
        u::Vector{Float64}
    )
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
                jac_g[k] = J[i, j]
                k += 1
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

function run_reduced_knitro(; eval_hessian=false, eval_cons=false)
    # datafile = "test/case9.m"
    # datafile = "test/case14.raw"
    # datafile = "/home/frapac/dev/pglib-opf/pglib_opf_case30_ieee.m"
    datafile = "/home/frapac/dev/pglib-opf/pglib_opf_case14_ieee.m"
    # datafile = "/home/frapac/dev/pglib-opf/pglib_opf_case300_ieee.m"
    # datafile = "/home/frapac/dev/pglib-opf/pglib_opf_case89_pegase.m"
    pf = PowerSystem.PowerNetwork(datafile, 1)

    # retrieve initial state of network
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)
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
    eval_f, eval_grad_f, eval_g, eval_jac_g, eval_hh = build_callback(pf, xk, uk, p)
    jac_g = zeros(n * npq)
    eval_jac_g(jac_g, uk)

    v_min = 0.9
    v_max = 1.1
    # Set bounds on decision variable
    x_L = zeros(n)
    x_U = zeros(n)
    # ... wrt. reference's voltages
    x_L[1:nref] .= v_min
    x_U[1:nref] .= v_max
    # ... wrt. active power in PV buses
    x_L[nref + 1:nref + npv] .= 0.0
    x_U[nref + 1:nref + npv] .= 3.0
    # ... wrt. voltages in PV buses
    x_L[nref + npv + 1:nref + 2*npv] .= v_min
    x_U[nref + npv + 1:nref + 2*npv] .= v_max

    # add constraint on PQ's voltage magnitude
    if eval_cons
        m = npq
        jnnz = m * n
        g_L = fill(v_min, m)
        g_U = fill(v_max, m)
        jac_rows = zeros(Cint, jnnz)
        jac_cols = zeros(Cint, jnnz)
        idx = 1
        for c in 1:npq #number of constraints
            for i in 1:n # number of variables
                jac_rows[idx] = c - 1; jac_cols[idx] = i - 1
                idx += 1
            end
        end
    else
        m = 0
        jnnz = 0
        g_L = Float64[]
        g_U = Float64[]
    end

    # Number of nonzeros in upper triangular Hessian
    if eval_hessian
        hnnz = div(n * (n+1), 2)
        eval_h = eval_hh
    else
        hnnz = 0
        eval_h = nothing
    end
    function kn_eval_f(kc, cb, evalRequest, evalResult, userParams)
        x = evalRequest.x
        evalResult.obj[1] = eval_f(x)
        if length(evalResult.c) > 0
            eval_g(x, evalResult.c)
        end
        return 0
    end
    function kn_eval_g(kc, cb, evalRequest, evalResult, userParams)
        x = evalRequest.x
        eval_grad_f(x, evalResult.objGrad)
        if length(evalResult.jac) > 0
            eval_jac_g(evalResult.jac, x)
        end
        return 0
    end

    kc = KNITRO.KN_new()
    iᵤ = KNITRO.KN_add_vars(kc, n)
    KNITRO.KN_set_var_lobnds(kc, iᵤ, x_L)
    KNITRO.KN_set_var_upbnds(kc, iᵤ, x_U)
    KNITRO.KN_set_var_primal_init_values(kc, iᵤ, uk)

    if eval_cons
        indexCons = KNITRO.KN_add_cons(kc, m)
        c_l = fill(v_min, m)
        c_u = fill(v_max, m)
        KNITRO.KN_set_con_lobnds(kc, indexCons, c_l)
        KNITRO.KN_set_con_upbnds(kc, indexCons, c_u)
        cb = KNITRO.KN_add_eval_callback(kc, true, indexCons, kn_eval_f)
        KNITRO.KN_set_cb_grad(kc, cb, kn_eval_g;
                              jacIndexCons=jac_rows,
                              jacIndexVars=jac_cols)
    else
        cb = KNITRO.KN_add_objective_callback(kc, kn_eval_f)
        KNITRO.KN_set_cb_grad(kc, cb, kn_eval_g)
    end

    KNITRO.KN_set_param(kc, "algorithm", 1)
    KNITRO.KN_set_param(kc, "hessopt", 2)
    # KNITRO.KN_set_param(kc, "derivcheck", 1)
    # KNITRO.KN_set_param(kc, "derivcheck_type", 2)

    KNITRO.KN_solve(kc)
    res = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    u = res[3]
    x, g, Jx, Ju, conv = ExaPF.solve(pf, xk, u, p, maxiter=50, tol=1e-14)
    ExaPF.PowerSystem.print_state(pf, x, u, p)
    return res
end

prob = run_reduced_knitro(;eval_hessian=false, eval_cons=true)
