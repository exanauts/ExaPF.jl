# Implementation of ideas found in F.F. Wu's "Two stage" paper.
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using Printf
using KernelAbstractions

import ExaPF: PowerSystem

# This function computes the Davidon cubic interpolation as seen in:
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.693.272&rep=rep1&type=pdf
# Notes:
#
#  - In some cases v^2 - F0p*FMp < 0. What to do if we need to keep alpha_m??
function davidon_ls(polar, xk, uk, p, delta_x, delta_u, alpha_m)

    F0 = ExaPF.cost_production(polar, xk, uk, p)
    FM = ExaPF.cost_production(polar, xk + alpha_m*delta_x, uk + alpha_m*delta_u, p)

    function cost_a(a)
        ud = uk + a*delta_u
        xkk, convergence = ExaPF.powerflow(polar, xk, ud, p)
        return ExaPF.cost_production(polar, xkk, ud, p)
    end

    F0p = FiniteDiff.finite_difference_derivative(cost_a, 0.0)
    FMp = FiniteDiff.finite_difference_derivative(cost_a, alpha_m)

    v = (3.0/alpha_m)*(F0 - FM) + F0p + FMp
    w = sqrt(v^2 - F0p*FMp)

    scale = (FMp + w - v)/(FMp - F0p + 2*w)
    alpha = alpha_m - scale*alpha_m

    return alpha
end

# sample along descent line and find minimum.
function sample_ls(nlp, uk, delta_x, delta_u, alpha_m; sample_max=30)

    xk = get(nlp, State())
    alpha = 0.0

    function cost_a(a)
        ud = uk + a*delta_u
        ExaPF.update!(nlp, ud)
        return ExaPF.objective(nlp, ud)
    end

    alpha_vec = collect(range(0, stop=alpha_m, length=sample_max))
    f_vec = zeros(sample_max)

    for i=1:sample_max
        a = alpha_vec[i]
        f_vec[i] = cost_a(a)
    end

    (val, ind) = findmin(f_vec)

    return alpha_vec[ind]
end

# This function computes an approximation of delta_x
# given delta_u
function deltax_approx(delta_u, dGdx, dGdu)
    b = -dGdu*delta_u
    delta_x = dGdx\b
    return delta_x
end

# This function determines the descent direction given gradient
# and limits on control variables.
function descent_direction(nlp, rk, u; damping=false, scale=2.0)
    pf = nlp.model.network
    u_min, u_max = nlp.u_min, nlp.u_max

    dim = length(u)
    delta_u = zeros(dim)
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

    for i=1:dim
        if u[i] < u_max[i] && u[i] > u_min[i]
            delta_u[i] = -rk[i]
        elseif isapprox(u[i], u_max[i]) && rk[i] > 0.0
            delta_u[i] = -rk[i]
        elseif isapprox(u[i], u_min[i]) && rk[i] < 0.0
            delta_u[i] = -rk[i]
        end
    end

    # u = [VMAG^{REF}, P^{PV}, VMAG^{PV}]

    # scale ratio
    for i=1:npv
        delta_u[nref + i] = scale*delta_u[nref + i]
    end

    # damping factor
    if damping
        for i=1:npv
            idx_u = nref + npv + i
            if u[idx_u] < u_max[idx_u] && u[idx_u] > u_min[idx_u]
                if rk[idx_u] < 0.0
                    damp = min((u_max[idx_u] - u[idx_u]), 1.0)
                    delta_u[idx_u] = damp*delta_u[idx_u]
                elseif rk[idx_u] > 0.0
                    damp = min((u[idx_u] - u_min[idx_u]), 1.0)
                    delta_u[idx_u] = damp*delta_u[idx_u]
                end
            end
        end
    end

    return delta_u
end

# This function determines if the algorithm has converged
function check_convergence(rk, u, u_min, u_max; eps=1e-4)
    dim = length(rk)

    for i=1:dim
        if u[i] < u_max[i] && u[i] > u_min[i] && abs(rk[i]) > eps
            return false
        elseif isapprox(u[i], u_max[i]) && rk[i] > eps
            return false
        elseif isapprox(u[i], u_min[i]) && rk[i] < eps
            return false
        end
    end

    return true
end

# obtain the maximum alpha along the descent line that satisfies all the
# constraints.
function alpha_max(nlp, delta_x, uk, delta_u)
    x_min, x_max = bounds(nlp.model, State())
    u_min, u_max = nlp.u_max, nlp.u_min
    xk = ExaPF.get(nlp, State())

    x_dim = length(delta_x)
    u_dim = length(delta_u)

    alpha_x = 1e10
    alpha_u = 1e10

    for i=1:u_dim
        if abs(delta_u[i]) > 0.0
            am = (u_max[i] - uk[i])/delta_u[i]
            al = (u_min[i] - uk[i])/delta_u[i]
            #@printf("i: %d. delta_u: %f. uk= %f, u_min=%f, u_max=%f, am=%f, al=%f\n",
            #        i, delta_u[i], uk[i], u_min[i], u_max[i], am, al)
            # alpha needs to be positive
            a_prop = max(am, al)
            # need to find alpha that satisfies all constraints
            alpha_u = min(alpha_u, a_prop)
        end
    end

    for i=1:x_dim
        if abs(delta_x[i]) > 0.0 && (x_max[i] > x_min[i])
            am = (x_max[i] - xk[i])/delta_x[i]
            al = (x_min[i] - xk[i])/delta_x[i]
            # alpha needs to be positive
            a_prop = max(am, al)
            # need to find alpha that satisfies all constraints
            alpha_x = min(alpha_x, a_prop)
        end
    end

    # in some cases there seems to be no feasible step with the computed
    # delta_x. maybe this is a problem with using linearized approximation.
    if alpha_x < 0.0
        return alpha_u
    end

    # TODO: we will only calculate alpha_max for u for now.
    # Need to take a closer look to delta_x approximation.
    return alpha_u
end

# given limit alpha, compute costs along a direction.
function cost_direction(nlp, u, delta_u, alpha_max, alpha_dav; points=10)

    alphas = zeros(points)
    costs = zeros(points)
    k = 1

    for a=range(0.0, stop=alpha_max, length=points)
        u_prop = u + a*delta_u
        ExaPF.update!(nlp, u_prop)
        c = ExaPF.objective(nlp, u_prop)

        alphas[k] = a
        costs[k] = c
        k += 1
    end

    #UNCOMMENT THESE LINES
    # plt = lineplot(alphas, costs, title = "Cost along alpha", width=80);
    # alpha_dav_vert = alpha_dav*ones(points)
    # scatterplot!(plt, alpha_dav_vert, costs)
    # println(plt)

end

function run_ffwu(datafile; bfgs=false)
    polar = PolarForm(datafile, CPU())

    xk = ExaPF.initial(polar, State())
    uk = ExaPF.initial(polar, Control())

    buffer = ExaPF.get(polar, ExaPF.PhysicalState())
    nlp = @time ExaPF.ReducedSpaceEvaluator(polar)

    # reduced gradient method
    iterations = 0
    iter_max = 30
    step = 0.0001
    converged = false
    norm_tol = 1e-5

    # initialize arrays
    grad = similar(uk)
    fill!(grad, 0)
    grad_prev = copy(grad)

    cost_history = zeros(iter_max)
    grad_history = zeros(iter_max)

    if bfgs
        H = InverseLBFGSOperator(Float64, length(uk), 50, scaling=true)
    end

    iter = 1
    @printf("%6s %12s %6s %6s\n", "iter", "objective", "αₘₐₓ", "αₗₛ")
    while !converged && iter <= iter_max
        # solve power flow and compute gradients
        ExaPF.update!(nlp, uk)

        # evaluate cost
        c = ExaPF.objective(nlp, uk)
        cost_history[iter] = c

        # check convergence
        converged = check_convergence(grad_prev, uk, nlp.u_min, nlp.u_max)

        # compute descent direction
        if bfgs
            delta_u = -H * grad_prev
        else
            ExaPF.gradient!(nlp, grad, uk)
            delta_u = descent_direction(nlp, grad, uk, damping=false, scale=2.0)
        end

        # compute gradients of G
        dGdx = nlp.state_jacobian.x.J
        dGdu = nlp.state_jacobian.u.J

        # Line search
        # 1st - compute approximate delta_x
        delta_x = deltax_approx(delta_u, dGdx, dGdu)
        # 2nd - compute a_m such that x + a_m*delta_x, u + a_m*delta_u
        # does not violate constraints
        a_m = alpha_max(nlp, delta_x, uk, delta_u)

        # 3rd - fit cubic and find its minimum.
        #a_dav = davidon_ls(pf, nlp.x, uk, p, delta_x, delta_u, a_m)
        a_ls = sample_ls(nlp, uk, delta_x, delta_u, a_m, sample_max=20)
        @printf("%6d %.8e %.6e %.6e %.4e\n", iter, c, a_m, a_ls, norm(delta_u))

        # Uncomment function below to plot the cost function along the descent direction
        # and the calculated alpha.
        # cost_direction(nlp, uk, delta_u, a_m, a_ls; points=100)

        # compute control step
        uk .= uk .+ a_ls*delta_u
        grad_history[iter] = norm(grad)

        # compute gradient
        if bfgs
            ExaPF.gradient!(nlp, grad, uk)
            bfgs && push!(H, a_ls * delta_u, grad - grad_prev)
            grad_prev .= grad
        end
        iter += 1
    end
    # uncomment to plot cost evolution
    # plt = lineplot(cost_history[1:iter - 1], title = "Cost history", width=80);
    # plt = lineplot(log10.(grad_history[1:iter - 1]), title = "Cost history", width=80);
    # println(plt)

    return
end
datafile = joinpath(dirname(@__FILE__), "..", "data", "case9.m")
run_ffwu(datafile; bfgs=false)
