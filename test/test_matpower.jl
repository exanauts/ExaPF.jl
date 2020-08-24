# Verify solutions against matpower results
using Test
using ExaPF

import ExaPF: ParseMAT, PowerSystem, IndexSet

@testset "Power flow 9 bus case" begin
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

    # test impedance matrix entries
    @test isapprox(real(pf.Ybus[1, 1]), 0.0)
    @test isapprox(imag(pf.Ybus[1, 1]), -17.3611111)

	S = [0.0000 + 0.0000im, 1.6300 + 0.0000im, 0.8500 + 0.0000im, 0.0000 + 0.0000im,
		-0.9000 - 0.3000im, 0.0000 + 0.0000im, -1.0000 - 0.3500im, 0.0000 + 0.0000im,
		-1.2500 - 0.5000im]

    @test isapprox(S, pf.sbus)
 
    println("Before solve")
    ExaPF.PowerSystem.print_state(pf, x, u, p)

    # solve power flow
    xk, J, Ju, convergence = ExaPF.solve(pf, x, u, p)
    
    println("Before solve")
    ExaPF.PowerSystem.print_state(pf, xk, u, p)

    x_sol = [0.9870068781579537, 0.9754722045044448, 1.003375449839184, 0.9856449067439345,
             0.996185273317625, 0.9576210937650547, -0.04200385129447893, -0.07011446830092488,
             0.033608106889679565, 0.010848015284769322, 0.06630715934781146, 
             -0.07592061900861094, 0.16875136481876485, 0.0832709533581424]
    
    @test isapprox(x_sol, xk, atol=1e-7)

    c = ExaPF.cost_function(pf, x, u, p)
    dCdx, dCdu = ExaPF.cost_gradients(pf, x, u, p)
end
