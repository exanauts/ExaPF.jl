
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using LinearOperators
using LineSearches
using Printf
using KernelAbstractions
using UnicodePlots
using Statistics

reldiff(a, b) = abs(a - b) / max(1, a)

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
    alpha, obj = algo(Lalpha, grad_Lalpha, Lgrad_Lalpha, 0.002, obj, dL_0)
    return alpha
end

# sample along descent line and find minimum.
function sample_ls(nlp, uk, d, alpha_m; sample_max=30)
    alpha = 0.0
    function cost_a(a)
        ud = uk + a*d
        try
            ExaPF.update!(nlp, ud)
            return ExaPF.objective(nlp, ud)
        catch
            return 1e20
        end
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
function dommel_method(datafile; bfgs=false, iter_max=200, itout_max=1)

    # Load problem.
    pf = ExaPF.PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, CPU())

    x0 = ExaPF.initial(polar, State())
    uk = ExaPF.initial(polar, Control())
    p = ExaPF.initial(polar, Parameters())
    u0 = copy(uk)

    buffer = ExaPF.get(polar, ExaPF.PhysicalState())
    constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
    nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, uk, p; constraints=constraints,
                                      ε_tol=1e-10)
    # Init a penalty evaluator with initial penalty c₀
    pen = ExaPF.PenaltyEvaluator(nlp, c₀=10.0)

    # initialize arrays
    grad = similar(uk)
    fill!(grad, 0)
    grad_prev = copy(grad)
    obj_prev = Inf

    cost_history = Float64[]
    grad_history = Float64[]

    if bfgs
        H = InverseLBFGSOperator(Float64, length(uk), 50, scaling=true)
        step = 1e-5
    else
        H = I
        step = 0.0001
    end

    for i_out in 1:itout_max
        iter = 1
        uk .= u0
        converged = false
        @printf("%6s %8s %4s %4s\n", "iter", "obj", "∇f", "αₗₛ")
        for i in 1:iter_max
            # solve power flow and compute gradients
            nlp.x .= x0
            ExaPF.update!(pen, uk)

            # evaluate cost
            c = ExaPF.objective(pen, uk)
            c_ref = ExaPF.objective(pen.nlp, uk)
            ExaPF.gradient!(pen, grad, uk)

            # compute control step
            # step = ls(ls_algo, pen, uk, c, grad)
            # step = ls(ls_algo, pen, uk, c, grad)
            # step = sample_ls(pen, uk, -grad, 0.00002; sample_max=10)
            # println(step)
            uk .= uk .- step * H * grad
            ExaPF.project_constraints!(pen.nlp, grad, uk)

            norm_grad = norm(grad)
            inf_pr = ExaPF.primal_infeasibility(pen.nlp, pen.cons)

            # check convergence
            if (iter%10 == 0)
                @printf("%6d %.6e %.3e %.2e %.2e %.2e\n", iter, c, c - c_ref, norm_grad, inf_pr, step)
            end
            iter += 1
            push!(grad_history, norm_grad)
            push!(cost_history, c)
            if bfgs
                push!(H, step * H * grad, grad .- grad_prev)
            end
            grad_prev .= grad
            # Check whether we have converged nicely
            if (norm_grad < 1e-2
                || (iter >= 4 && reldiff(c, mean(cost_history[end-2:end])) < 1e-8)
               )
                converged = true
                break
            end
        end
        # Update penalty term, according to Nocedal & Wright §17.1 (p.501)
        # Safeguard: update nicely the penalty if previously we failed to converge
        η = converged ? 10.0 : 1.5
        @printf("Outer it %d \t obj: %.4e\n", i_out, ExaPF.objective(pen.nlp, uk))
        ExaPF.update_penalty!(pen; η=η)
    end
    # uncomment to plot cost evolution
    plt = lineplot(cost_history, title = "Cost history", width=80);
    # plt = lineplot(log10.(grad_history[1:iter - 1]), title = "Cost history", width=80);
    println(plt)

    n_cons = ExaPF.n_constraints(nlp)
    cons = zeros(n_cons)
    ExaPF.constraint!(nlp, cons, uk)
    ExaPF.sanity_check(nlp, uk, cons)

    return
end

datafile = joinpath(dirname(@__FILE__), "..", "test", "data", "case57.m")
#
dommel_method(datafile; bfgs=true, itout_max=5)
