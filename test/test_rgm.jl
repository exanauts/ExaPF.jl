# Verify solutions against matpower results
using Test
using ExaPF
using FiniteDiff
using ForwardDiff

import ExaPF: ParseMAT, PowerSystem, IndexSet

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

    # solve power flow
    xk, dGdx, dGdu, convergence = ExaPF.solve(pf, x, u, p)

    c = ExaPF.cost_function(pf, xk, u, p)
    dCdx, dCdu = ExaPF.cost_gradients(pf, xk, u, p)

    function cost_x(xk)
        return ExaPF.cost_function(pf, xk, u, p; V=eltype(xk))
    end
    
    function cost_u(uk)
        return ExaPF.cost_function(pf, xk, uk, p; V=eltype(uk))
    end
    
    println("dCdx")
    dCdx_fd = FiniteDiff.finite_difference_gradient(cost_x,xk)
    dCdx_ad = ForwardDiff.gradient(cost_x,xk)
    println("dCdu")
    dCdu_fd = FiniteDiff.finite_difference_gradient(cost_u,u)
    dCdu_ad = ForwardDiff.gradient(cost_u,u)
    
    @test isapprox(dCdx,dCdx_fd)
    @test isapprox(dCdu,dCdu_fd)
    @test isapprox(dCdx,dCdx_ad)
    @test isapprox(dCdu,dCdu_ad)
end
