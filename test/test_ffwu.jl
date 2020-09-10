# Implementation of ideas found in F.F. Wu's "Two stage" paper.
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using Printf
using KernelAbstractions
#using UnicodePlots

# Include the linesearch here for now
import ExaPF: ParseMAT, PowerSystem, IndexSet

# This function computes the Davidon cubic interpolation as seen in:
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.693.272&rep=rep1&type=pdf
# Notes:
#
#  - In some cases v^2 - F0p*FMp < 0. What to do if we need to keep alpha_m??
function davidon_ls(pf, xk, uk, p, delta_x, delta_u, alpha_m)

    F0 = ExaPF.cost_function(pf, xk, uk, p)
    FM = ExaPF.cost_function(pf, xk + alpha_m*delta_x, uk + alpha_m*delta_u, p)

    function cost_a(a)
        xd = xk + a*delta_x
        ud = uk + a*delta_u
        return ExaPF.cost_function(pf, xd, ud, p; V=eltype(xk))
    end

    F0p = FiniteDiff.finite_difference_derivative(cost_a, 0.0)
    FMp = FiniteDiff.finite_difference_derivative(cost_a, alpha_m)

    v = (3.0/alpha_m)*(F0 - FM) + F0p + FMp
    w = sqrt(v^2 - F0p*FMp)

    scale = (FMp + w - v)/(FMp - F0p + 2*w)
    alpha = alpha_m - scale*alpha_m

    return alpha
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
function descent_direction(pf, rk, u, u_min, u_max; damping=false)

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
    scale = 2.0
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

function alpha_max(xk, delta_x, uk, delta_u, x_min, x_max, u_min, u_max)

    x_dim = length(delta_x)
    u_dim = length(delta_u)

    alpha_x = 1e10
    alpha_u = 1e10

    for i=1:u_dim
        if abs(delta_u[i]) > 0.0
            am = (u_max[i] - uk[i])/delta_u[i]
            al = (u_min[i] - uk[i])/delta_u[i]
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
    return min(alpha_x, alpha_u)
end

# given limit alpha, compute costs along a direction.
function cost_direction(pf, x, u, p, delta_u, alpha_max, alpha_dav; points=10)

    alphas = zeros(points)
    costs = zeros(points)
    k = 1

    for a=range(0.0, stop=alpha_max, length=points)
        u_prop = u + a*delta_u
        xk, g, Jx, Ju, convergence = ExaPF.solve(pf, x, u_prop, p)
        c = ExaPF.cost_function(pf, xk, u_prop, p; V=eltype(xk))

        alphas[k] = a
        costs[k] = c
        k += 1
    end
    
    # UNCOMMENT THESE LINES
    #plt = lineplot(alphas, costs, title = "Cost along alpha", width=80);
    #alpha_dav_vert = alpha_dav*ones(points)
    #scatterplot!(plt, alpha_dav_vert, costs)
    #println(plt)

end

@testset "Two-stage OPF" begin
    datafile = joinpath(dirname(@__FILE__), "data", "case9.m")
    pf = PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, CPU())
    
    xk = ExaPF.initial(polar, State())
    uk = ExaPF.initial(polar, Control())
    p = ExaPF.initial(polar, Parameters())
    
    constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
    nlp = @time ExaPF.ReducedSpaceEvaluator(polar, xk, uk, p; constraints=constraints)
    jx, ju = ExaPF.init_ad_factory(polar, xk, uk, p)

    # reduced gradient method
    iterations = 0
    iter_max = 4
    step = 0.0001
    converged = false
    norm_tol = 1e-5

    # initialize arrays
    grad = similar(uk)
    fill!(grad, 0)
    cost_history = zeros(iter_max)
    grad_history = zeros(iter_max)

    # retrieve bounds
    u_min, u_max, x_min, x_max = ExaPF.PowerSystem.get_bound_constraints(pf)

    iter = 1
    while !converged && iter < iter_max
        println("Iteration: ", iter)
        # solve power flow and compute gradients
        ExaPF.update!(nlp, uk)

        # evaluate cost
        c = ExaPF.objective(nlp, uk)
        cost_history[iter] = c
        println("Cost: ", c)

        # compute gradient
        ExaPF.gradient!(nlp, grad, uk)
        println("Gradient norm: ", norm(grad))

        # check convergence
        converged = check_convergence(grad, uk, u_min, u_max)

        # compute descent direction
        delta_u = descent_direction(pf, grad, uk, u_min, u_max)

        # compute gradients of G
        dGdx = ExaPF.jacobian(polar, jx, nlp.x, uk, p)
        dGdu = ExaPF.jacobian(polar, ju, nlp.x, uk, p)

        # Line search
        # 1st - compute approximate delta_x
        delta_x = deltax_approx(delta_u, dGdx, dGdu)
        # 2nd - compute a_m such that x + a_m*delta_x, u + a_m*delta_u
        # does not violate constraints
        a_m = alpha_max(nlp.x, delta_x, uk, delta_u, x_min, x_max, u_min, u_max)
        # 3rd - fit cubic and find its minimum.
        a_dav = davidon_ls(pf, nlp.x, uk, p, delta_x, delta_u, a_m)

        @printf("Maximal alpha: %f\n", a_m)
        @printf("Davidon alpha: %f\n", a_dav)
        
        # Uncomment function below to plot the cost function along the descent direction
        # and the calculated alpha.
        #cost_direction(pf, nlp.x, uk, p, delta_u, a_m, a_dav; points=10)
        
        # compute control step
        uk = uk + a_dav*delta_u

        iter += 1
    end
    ExaPF.PowerSystem.print_state(pf, nlp.x, uk, p)
    
    # uncomment to plot cost evolution
    #plt = lineplot(cost_history[1:iter - 1], title = "Cost history", width=80);
    #println(plt)

    return
end
