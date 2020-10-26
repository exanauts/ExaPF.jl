
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

