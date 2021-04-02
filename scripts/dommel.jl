
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using Printf
using KernelAbstractions
using Statistics

reldiff(a, b) = abs(a - b) / max(1, a)
function active_set(x, x♭, x♯; tol=1e-8)
    are_min = findall(x .<= x♭ .+ tol)
    are_max = findall(x .>= x♯ .- tol)
    return vcat(are_min, are_max)
end

function ls(algo, nlp, uk::Vector{Float64}, obj, grad::Vector{Float64})
    nᵤ = length(grad)
    s = copy(-grad)
    function Lalpha(alpha)
        u_ = uk .+ alpha.*s
        ExaPF.update!(nlp, u_)
        return ExaPF.objective(nlp, u_)
    end
    function grad_Lalpha(alpha)
        g_ = zeros(nᵤ)
        u_ = uk .+ alpha .* s
        ExaPF.update!(nlp, u_)
        ExaPF.gradient!(nlp, g_, u_)
        return dot(g_, s)
    end
    function Lgrad_Lalpha(alpha)
        g_ = zeros(nᵤ)
        u_ = uk .+ alpha .* s
        ExaPF.update!(nlp, u_)
        ExaPF.gradient!(nlp, g_, u_)
        phi = ExaPF.objective(nlp, u_)
        dphi = dot(g_, s)
        return (phi, dphi)
    end
    dL_0 = dot(s, grad)
    alpha, obj = algo(Lalpha, grad_Lalpha, Lgrad_Lalpha, 0.0002, obj, dL_0)
    return alpha
end

# sample along descent line and find minimum.
function sample_ls(nlp, uk, d, alpha_m; sample_max=30)
    alpha = 0.0
    function cost_a(a)
        ud = uk + a*d
        ExaPF.update!(nlp, ud)
        return ExaPF.objective(nlp, ud)
    end

    alpha_vec = collect(range(0.1*alpha_m, stop=alpha_m, length=sample_max))
    f_vec = zeros(sample_max)

    for i=1:sample_max
        a = alpha_vec[i]
        f_vec[i] = cost_a(a)
    end

    (val, ind) = findmin(f_vec)

    return alpha_vec[ind]
end

# reduced gradient method
function dommel_method(datafile; bfgs=false, iter_max=200, itout_max=1,
                       feasible_start=false)

    # Load problem.
    polar = PolarForm(datafile, CPU())

    x0 = ExaPF.initial(polar, State())
    if feasible_start
        prob = run_reduced_ipopt(datafile; hessian=false, feasible=true)
        uk = prob.x
    else
        uk = ExaPF.initial(polar, Control())
        # uk = nlp.u_min
    end
    u0 = copy(uk)
    u_start = copy(u0)
    wk = copy(uk)
    u_prev = copy(uk)

    buffer = ExaPF.get(polar, ExaPF.PhysicalState())
    nlp = ExaPF.ReducedSpaceEvaluator(polar;
                                      powerflow_solver=NewtonRaphson(; tol=1e-10))
    # Init a penalty evaluator with initial penalty c₀
    c0 = 10.0

    pen = ExaPF.AugLagEvaluator(nlp, u0; c₀=c0, scale=true)
    ωtol = 1e-5 #1 / c0

    # initialize arrays
    grad = similar(uk)
    # ut for line search
    ut = similar(uk)
    fill!(grad, 0)
    grad_prev = copy(grad)
    obj_prev = Inf
    norm_grad = Inf

    outer_costs = Float64[]
    cost_history = Float64[]
    grad_history = Float64[]

    if bfgs
        H = InverseLBFGSOperator(Float64, length(uk), 50, scaling=true)
        α0 = 1.0
    else
        H = I
        α0 = 1e-6 # REF
        # α0 = 1e-4
    end
    αi = α0
    u♭ = nlp.u_min
    u♯ = nlp.u_max
    ls_itermax = 30
    β = 0.4
    τ = 1e-4

    for i_out in 1:itout_max
        iter = 1
        uk .= u_start
        converged = false
        # @printf("%6s %8s %4s %4s\n", "iter", "obj", "∇f", "αₗₛ")
        # Inner iteration: projected gradient algorithm
        n_iter = 0
        for i in 1:iter_max
            n_iter += 1
            # solve power flow and compute gradients
            ExaPF.update!(pen, uk)

            # evaluate cost
            c = ExaPF.objective(pen, uk)
            # Evaluate cost of problem without penalties
            c_ref = pen.scaler.scale_obj * ExaPF.objective(pen.inner, uk)
            ExaPF.gradient!(pen, grad, uk)

            # compute control step
            # Armijo line-search (Bertsekas, 1976)
            dk = H * grad
            step = αi
            for j_ls in 1:ls_itermax
                step *= β
                ExaPF.project!(ut, uk .- step .* dk, u♭, u♯)
                ExaPF.update!(pen, ut)
                ft = ExaPF.objective(pen, ut)
                if ft <= c - τ * dot(dk, ut .- uk)
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
            inf_pr = ExaPF.primal_infeasibility(pen.inner, pen.cons)

            # check convergence
            if (iter%100 == 0)
                @printf("%6d %.6e %.3e %.2e %.2e %.2e\n", i, c, c - c_ref, norm_grad, inf_pr, step)
            end
            iter += 1
            push!(grad_history, norm_grad)
            push!(cost_history, c)
            if bfgs
                push!(H, uk .- u_prev, grad .- grad_prev)
            end
            grad_prev .= grad
            u_prev .= uk
            # Check whether we have converged nicely
            if (norm_grad < ωtol
                || (iter >= 4 && reldiff(c, mean(cost_history[end-2:end])) < 1e-7)
            )
                converged = true
                break
            end
        end
        # Update penalty term, according to Nocedal & Wright §17.1 (p.501)
        # Safeguard: update nicely the penalty if previously we failed to converge
        if converged
            ρ = 1e-6
            η = 10.0
        else
            ρ = 1e-6
            η = 2.0
        end
        # αi = max(2.0 / η * αi, 1e-8)
        u_start .= ρ * u0 .+ (1 - ρ) .* uk
        ωtol *= 1 / η
        ωtol = max(ωtol, 1e-6)

        # Evaluate current position in the original space
        cons = zeros(ExaPF.n_constraints(nlp))
        ExaPF.constraint!(nlp, cons, uk)
        obj = ExaPF.objective(nlp, uk)
        inf_pr = ExaPF.primal_infeasibility(nlp, cons)
        push!(outer_costs, obj)
        @printf("#Outer %d %-4d %.3e %.3e \n",
                i_out, n_iter, obj, inf_pr)
        if (norm_grad < 1e-6) && (inf_pr < 1e-8)
            break
        end
        ExaPF.update_penalty!(pen; η=η)
    end
    # uncomment to plot cost evolution
    # plt = lineplot(cost_history, title = "Cost history", width=80);
    # println(plt)

    cons = zeros(ExaPF.n_constraints(nlp))
    ExaPF.constraint!(nlp, cons, uk)
    ExaPF.sanity_check(nlp, uk, cons)

    return uk, cost_history
end

datafile = joinpath(dirname(@__FILE__), "..", "data", "case57.m")

u_opt, ch = dommel_method(datafile; bfgs=false, itout_max=10, feasible_start=false,
                          iter_max=1000)
