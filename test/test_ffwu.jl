# Verify solutions against matpower results
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using LinearAlgebra

# Include the linesearch here for now
import ExaPF: ParseMAT, PowerSystem, IndexSet


function deltax_approx(delta_u, dGdx, dGdu)
    b = -dGdu*delta_u
    delta_x = dGdx\b
    return delta_x
end

function descent_direction(rk, u, u_min, u_max)

    dim = length(u)
    delta_u = zeros(dim)

    for i=1:dim
        if u[i] < u_max[i] && u[i] > u_min[i]
            delta_u[i] = -rk[i]
        elseif isapprox(u[i], u_max[i]) && rk[i] > 0.0
            delta_u[i] = -rk[i]
        elseif isapprox(u[i], u_min[i]) && rk[i] < 0.0
            delta_u[i] = -rk[i]
        end
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


@testset "RGM Optimal Power flow 9 bus case" begin
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
    iter_max = 100
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
        delta_u = descent_direction(grad, uk, u_min, u_max)

        # line search
        delta_x = deltax_approx(delta_u, dGdx, dGdu)

        # compute control step
        uk = uk + step*delta_u
        #ExaPF.project_constraints!(uk, grad, u_min, u_max)
        println("Gradient norm: ", norm(grad))
        norm_grad = norm(grad)

        iter += 1
    end
    ExaPF.PowerSystem.print_state(pf, xk, uk, p)
 

end
