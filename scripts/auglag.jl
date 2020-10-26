
using ExaPF
using LinearAlgebra
using LineSearches
using Printf
using KernelAbstractions
using UnicodePlots
using Statistics

include("algo.jl")

# Augmented Lagrangian method
function auglag(datafile; bfgs=false, iter_max=200, itout_max=1,
                       feasible_start=false)

    # Load problem.
    pf = ExaPF.PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, CPU())

    x0 = ExaPF.initial(polar, State())
    p = ExaPF.initial(polar, Parameters())
    if feasible_start
        prob = run_reduced_ipopt(datafile; hessian=false, feasible=true)
        uk = prob.x
    else
        uk = ExaPF.initial(polar, Control())
    end
    u0 = copy(uk)
    u_start = copy(u0)
    wk = copy(uk)
    u_prev = copy(uk)

    buffer = ExaPF.get(polar, ExaPF.PhysicalState())
    constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
    nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, uk, p; constraints=constraints,
                                      ε_tol=1e-10)
    # Init a penalty evaluator with initial penalty c₀
    c0 = 0.1

    aug = ExaPF.AugLagEvaluator(nlp, u0; c₀=c0, scale=true)
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
    alpha_history = Float64[]

    ls_algo = BackTracking()
    if bfgs
        H = InverseLBFGSOperator(Float64, length(uk), 50, scaling=true)
        α0 = 1.0
    else
        H = I
        α0 = 1e-3
    end
    αi = α0
    u♭ = nlp.u_min
    u♯ = nlp.u_max
    ηk = 1.0 / (c0^0.1)

    for i_out in 1:itout_max
        iter = 1
        uk .= u_start
        converged = false
        # Inner iteration: projected gradient algorithm
        uk, norm_grad, n_iter = projected_gradient(aug, uk; α0=α0)

        # Evaluate current position in the original space
        cons = zeros(ExaPF.n_constraints(nlp))
        ExaPF.constraint!(nlp, cons, uk)
        obj = ExaPF.objective(nlp, uk)
        inf_pr = ExaPF.primal_infeasibility(nlp, cons)
        push!(outer_costs, obj)
        @printf("#Outer %d %-4d %.3e %.3e \n", i_out, n_iter, obj, inf_pr)

        if (norm_grad < 1e-6) && (inf_pr < 1e-8)
            break
        end

        ρ = 0e-6
        u_start .= ρ * u0 .+ (1 - ρ) .* uk

        # Update the parameters (see Nocedal & Wright, page 521)
        if norm(abs.(aug.infeasibility), Inf) <= ηk
            ExaPF.update_multipliers!(aug)
            ηk = ηk / (aug.ρ^0.9)
        else
            ExaPF.update_penalty!(aug; η=10.0)
            ηk = 1.0 / (aug.ρ^0.1)
        end
    end
    # uncomment to plot cost evolution
    # plt = lineplot(log10.(cost_history), title = "Cost history", width=80);
    # println(plt)

    cons = zeros(ExaPF.n_constraints(nlp))
    ExaPF.constraint!(nlp, cons, uk)
    ExaPF.sanity_check(nlp, uk, cons)

    return uk, cost_history
end

datafile = joinpath(dirname(@__FILE__), "..", "test", "data", "case57.m")

u_opt, ch = auglag(datafile; bfgs=false, itout_max=10, feasible_start=false,
                   iter_max=1000)

