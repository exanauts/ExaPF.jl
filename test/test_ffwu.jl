# Implementation of ideas found in F.F. Wu's "Two stage" paper.
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using Printf
using UnicodePlots

# Include the linesearch here for now
import ExaPF: ParseMAT, PowerSystem, IndexSet


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


function deltax_approx(delta_u, dGdx, dGdu)
    b = -dGdu*delta_u
    delta_x = dGdx\b
    return delta_x
end

function descent_direction(pf, rk, u, u_min, u_max)

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
    scale = 2.0
    for i=1:npv
        delta_u[nref + i] = scale*delta_u[nref + i]
    end

    return delta_u
end

function check_convergence(rk, u, u_min, u_max; eps=1e-5)
    
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
            if a_prop < 0.0
                println("Alpha_x negative!")
            end
            # need to find alpha that satisfies all constraints
            alpha_x = min(alpha_x, a_prop)
        end
    end
    return min(alpha_x, alpha_u)
end

# given limit alpha, compute costs along a direction.

function cost_direction(pf, x, u, p, delta_u, alpha_max; points=10)

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
    plt = lineplot(alphas, costs, title = "Cost along alpha", width=80);
    println(plt)

end


@testset "Two-stage OPF" begin
    datafile = "test/case9.m"
    pf = PowerSystem.PowerNetwork(datafile, 1)

    # retrieve initial state of network
    pbus = real.(pf.sbus)
    qbus = imag.(pf.sbus)
    vmag = abs.(pf.vbus)
    vang = angle.(pf.vbus)

    x = ExaPF.PowerSystem.get_x(pf, vmag, vang, pbus, qbus)
    u = ExaPF.PowerSystem.get_u(pf, vmag, vang, pbus, qbus)
    p = ExaPF.PowerSystem.get_p(pf, vmag, vang, pbus, qbus)
    u_min, u_max, x_min, x_max = ExaPF.get_constraints(pf)

    # solve power flow
    xk, g, Jx, Ju, convergence = ExaPF.solve(pf, x, u, p)
    dGdx = Jx(pf, x, u, p)
    dGdu = Ju(pf, x, u, p)

    c = ExaPF.cost_function(pf, xk, u, p)
    dCdx, dCdu = ExaPF.cost_gradients(pf, xk, u, p)

    uk = copy(u)

    # reduced gradient method
    iterations = 0
    iter_max = 3
    step = 0.0001
    norm_grad = 10000
    converged = false
    norm_tol = 1e-5

    cost_history = zeros(iter_max)
    grad_history = zeros(iter_max)

    iter = 1
    while !converged && iter < iter_max
        println("Iteration: ", iter)
        # solve power flow and compute gradients
        xk, g, Jx, Ju, convergence = ExaPF.solve(pf, xk, uk, p)
        dGdx = Jx(pf, xk, uk, p)
        dGdu = Ju(pf, xk, uk, p)

        # evaluate cost
        c = ExaPF.cost_function(pf, xk, uk, p; V=eltype(xk))
        dCdx, dCdu = ExaPF.cost_gradients(pf, xk, uk, p)
        
        # lamba calculation
        lambda = -(dGdx'\dCdx)
        
        # Compute gradient
        grad = dCdu + (dGdu')*lambda
        println("Cost: ", c)
        println("Norm: ", norm(grad))
 
        # check convergence
        converged = check_convergence(grad, uk, u_min, u_max)

        # compute descent direction
        delta_u = descent_direction(pf, grad, uk, u_min, u_max)

        # line search
        delta_x = deltax_approx(delta_u, dGdx, dGdu)
        a_m = alpha_max(xk, delta_x, uk, delta_u, x_min, x_max, u_min, u_max)
        @printf("Maximal alpha: %f\n", a_m)
        a_dav = davidon_ls(pf, xk, uk, p, delta_x, delta_u, a_m)
        @printf("Davidon alpha: %f\n", a_dav)

        cost_direction(pf, xk, uk, p, delta_u, a_m; points=10)
        
        # compute control step
        uk = uk + step*delta_u
        
        println("Gradient norm: ", norm(grad))
        norm_grad = norm(grad)

        println(grad)
        iter += 1
    end
    ExaPF.PowerSystem.print_state(pf, xk, uk, p)
 

end
