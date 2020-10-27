
function projected_gradient(
    nlp,
    uk;
    max_iter=1000,
    ls_itermax=30,
    α0=1.0,
    β=0.4,
    τ=1e-4,
    tol=1e-5,
    bfgs=false,
    verbose_it=100,
)

    u_prev = copy(uk)
    grad = copy(uk)
    wk = copy(uk)
    u♭ = nlp.inner.u_min
    u♯ = nlp.inner.u_max
    if bfgs
        H = InverseLBFGSOperator(Float64, length(uk), 50, scaling=true)
    else
        H = I
    end
    norm_grad = Inf
    n_iter = 0

    for i in 1:max_iter
        n_iter += 1
        # solve power flow and compute gradients
        ExaPF.update!(nlp, uk)

        # evaluate cost
        c = ExaPF.objective(nlp, uk)
        # Evaluate cost of problem without penalties
        c_ref = nlp.scaler.scale_obj * ExaPF.objective(nlp.inner, uk)
        ExaPF.gradient!(nlp, grad, uk)

        # compute control step
        # Armijo line-search (Bertsekas, 1976)
        dk = H * grad
        step = α0
        for j_ls in 1:ls_itermax
            step *= β
            ExaPF.project!(wk, uk .- step .* dk, u♭, u♯)
            ExaPF.update!(nlp, wk)
            ft = ExaPF.objective(nlp, wk)
            if ft <= c - τ * dot(dk, wk .- uk)
                break
            end
        end

        # step = αi
        wk .= uk .- step * dk
        ExaPF.project!(uk, wk, u♭, u♯)

        # Stopping criteration: uₖ₊₁ - uₖ
        ## Dual infeasibility
        norm_grad = norm(uk .- u_prev, Inf)
        ## Primal infeasibility
        inf_pr = ExaPF.primal_infeasibility(nlp.inner, nlp.cons)

        # check convergence
        if (i % verbose_it == 0)
            @printf("%6d %.6e %.3e %.2e %.2e %.2e\n", i, c, c - c_ref, norm_grad, inf_pr, step)
        end

        if bfgs
            push!(H, uk .- u_prev, grad .- grad_prev)
            grad_prev .= grad
        end
        u_prev .= uk
        # Check whether we have converged nicely
        if (norm_grad < tol)
            converged = true
            break
        end
    end
    return uk, norm_grad, n_iter
end


function ls_reference(costs::AbstractVector, lm_size::Int)
    lag = max(1, length(costs) - lm_size + 1)
    return maximum(costs[lag:end])
end
# Non-monotone gradient projection algorithm
function ngpa(
    nlp,
    uk;
    max_iter=1000,
    ls_itermax=30,
    α_bb=1.0,
    α♭=0.0,
    α♯=1.0,
    β=0.4,
    δ=1e-4,
    tol=1e-5,
    lm_size=1,
    verbose_it=100,
)

    u_prev = copy(uk)
    grad = copy(uk)
    wk = copy(uk)
    u♭ = nlp.inner.u_min
    u♯ = nlp.inner.u_max

    # Initial evaluation
    ExaPF.update!(nlp, uk)
    c = ExaPF.objective(nlp, uk)
    ExaPF.gradient!(nlp, grad, uk)

    # Memory
    grad_prev = copy(grad)
    inf_pr = ExaPF.primal_infeasibility(nlp.inner, nlp.cons ./ nlp.scaler.scale_cons)
    costs_history = Float64[c]
    grad_history = Float64[norm(grad, Inf)]
    feas_history = Float64[inf_pr]

    norm_grad = Inf
    n_iter = 0
    ## Line-search params
    j_bb = 0
    flag_bb = 1
    θ_bb = 0.975
    m_bb = 10
    ## Reference function params
    Δ_ref = 0.1
    A_ref = 0
    L_ref = 0
    a_ref = 0
    l_ref = 0
    fᵣ = c

    n_up = 0
    for i in 1:max_iter
        n_iter += 1

        ExaPF.project!(wk, uk .- α_bb .* grad, u♭, u♯)
        # Feasible direction
        dk = wk .- uk
        # Armijo line-search
        step = 1.0
        for j_ls in 1:ls_itermax
            ExaPF.project!(wk, uk .+ step .* dk, u♭, u♯)
            conv = ExaPF.update!(nlp, wk)
            ft = ExaPF.objective(nlp, wk)
            if ft <= fᵣ + step * δ * dot(dk, grad)
                break
            end
            step *= β
        end

        uk .= wk
        # Objective
        c = ExaPF.objective(nlp, uk)
        c_ref = ExaPF.inner_objective(nlp, uk)
        # Gradient
        ExaPF.gradient!(nlp, grad, uk)

        # Stopping criteration: uₖ₊₁ - uₖ
        ## Dual infeasibility
        norm_grad = norm(uk .- u_prev, Inf)
        ## Primal infeasibility
        inf_pr = ExaPF.primal_infeasibility(nlp.inner, nlp.cons ./ nlp.scaler.scale_cons)

        # check convergence
        if (i % verbose_it == 0)
            @printf("%6d %.6e %.3e %.2e %.2e %.2e\n", i, c, c - c_ref, norm_grad, inf_pr, step)
        end

        ##################################################
        ## Update parameters
        sk = uk - u_prev
        yk = grad - grad_prev
        ##################################################
        ## Update Barzilai-Borwein step
        flag_bb = 0
        if !isnothing(findfirst(0.0 .< abs.(dk) .< α_bb .* abs.(grad)))
            flag_bb = 1
        end
        if step == 1.0
            j_bb += 1
        else
            flag_bb = 1
        end
        θ = dot(sk, yk) / (norm(sk) * norm(yk))
        if (j_bb >= m_bb) || (θ >= θ_bb) || (flag_bb == 1)
            # Non-convexity detected
            if dot(sk, yk) <= 0.0
                if j_bb >= 1.5 * m_bb
                    t_bb = min(norm(uk, Inf), 1) / norm(dk, Inf)
                    α_bb = max(t, step)
                    j_bb = 0
                end
            else
                # Everything went ok. Set new step to Barzilai-Borwein value
                α_bb = dot(sk, sk) / dot(sk, yk)
                j_bb = 0
            end
        end
        α_bb = min(α♯, α_bb)
        # Update history
        u_prev .= uk
        grad_prev .= grad
        push!(costs_history, c)
        push!(grad_history, norm_grad)
        push!(feas_history, inf_pr)
        ##################################################
        ## Update reference value
        fr_max = ls_reference(costs_history, lm_size)
        fᵣ = .5 * (fr_max + c)
        # if j > 0
        #     fᵣ = min(fr_max, fᵣ)
        # end
        # a_ref = (step < 1) ? 0 : a_ref + 1
        # if kkkk

        # Check whether we have converged nicely
        if (norm_grad < tol)
            converged = true
            break
        end
    end
    return uk, norm_grad, n_iter, costs_history, grad_history, feas_history
end

