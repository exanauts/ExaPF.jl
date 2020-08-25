# Verify solutions against matpower results
using Test
using ExaPF

import ExaPF: ParseMAT, PowerSystem, IndexSet

@testset "Power flow 9 bus case" begin
    datafile = "case9.m"
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

@testset "Power flow 14 bus case" begin
    datafile = "case14.m"
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
    xk, g, Jx, Ju, conv, f = ExaPF.solve(pf, x, u, p)
    
    @test conv.n_iterations == 2
    @test isapprox(conv.norm_residuals, 1.3158e-10, rtol=1e-4)
    
    vmag, vang, pinj, qinj = PowerSystem.retrieve_physics(pf, xk, u, p)
    vmag_matpower = [1.060000,1.045000,1.010000,1.017671,1.019514,1.070000,1.061520,1.090000,1.055932,1.050985,1.056907,1.055189,1.050382,1.035530]

    @test isapprox(vmag, vmag_matpower, rtol=1e-6)
end

@testset "Power flow 30 bus case" begin
    datafile = "case30.m"
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
    xk, g, Jx, Ju, conv, f = ExaPF.solve(pf, x, u, p)
    
    @test conv.n_iterations == 3
    @test isapprox(conv.norm_residuals, 9.56998e-10, rtol=1e-4)
    
    vmag, vang, pinj, qinj = PowerSystem.retrieve_physics(pf, xk, u, p)

    vmag_matpower = [1.000000,1.000000,0.983138,0.980093,0.982406,0.973184,0.967355,0.960624,0.980506,0.984404,0.980506,0.985468,1.000000,0.976677,0.980229,0.977396,0.976865,0.968440,0.965287,0.969166,0.993383,1.000000,1.000000,0.988566,0.990215,0.972194,1.000000,0.974715,0.979597,0.967883]

    @test isapprox(vmag, vmag_matpower, rtol=1e-6)
end

@testset "Power flow 300 bus case" begin
    datafile = "case300.m"
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
    xk, g, Jx, Ju, conv, f = ExaPF.solve(pf, x, u, p)
    
    @test conv.n_iterations == 5
    @test isapprox(conv.norm_residuals, 1.3783e-12, rtol=1e-2)
    
    vmag, vang, pinj, qinj = PowerSystem.retrieve_physics(pf, xk, u, p)
    vmag_matpower = [1.028420,1.035340,0.997099,1.030812,1.019109,1.031196,0.993408,1.015300,1.003386,1.020500,1.005657,0.997373,0.997674,0.999170,1.034391,1.031587,1.064906,0.981924,1.001000,0.975168,0.996270,1.050098,1.005656,1.023354,0.998577,0.975035,1.024565,1.041441,0.975688,1.001170,1.020158,1.020312,1.053611,1.021655,1.029283,1.044944,1.000732,1.008735,1.021646,1.034515,0.977877,1.001958,1.047499,1.025388,0.998003,0.996035,1.005135,1.015146,1.033490,0.991822,0.978860,1.024716,0.990654,1.016040,0.958300,0.947956,0.962698,0.951318,0.979391,0.969614,0.977610,0.996488,0.963200,0.983787,0.990023,0.982012,0.987242,1.034127,1.025000,0.987112,0.990818,0.991954,1.015248,1.031724,1.027231,1.052000,1.052000,0.992945,1.018224,1.000000,0.989358,1.005986,1.000708,1.028759,0.995737,1.022267,1.009461,0.990000,0.975245,0.973213,0.974473,0.970155,0.976812,0.960282,1.024861,0.934829,0.929853,1.043500,0.958437,0.987111,0.972796,1.000588,1.023300,1.010300,0.997795,1.000129,1.002406,1.002825,1.019136,0.986142,1.004551,1.001998,1.022076,1.019337,1.047586,1.047088,1.055000,1.011709,1.042991,1.051000,1.015510,1.043500,1.016107,1.008106,1.052800,1.052800,1.057719,1.073500,0.986926,1.004833,1.053500,1.043500,0.966417,1.017724,0.963000,0.984473,0.998709,0.986644,0.999801,1.036082,0.991820,1.041011,0.983914,1.000211,0.997254,0.971492,1.002431,0.987864,0.929000,0.982900,1.024466,0.983654,1.062214,0.973081,1.052200,1.007700,0.939796,0.969910,0.979330,1.051824,1.044628,0.971645,1.038589,1.052200,1.065000,1.065000,1.053282,1.002757,1.055100,1.043500,0.937458,0.998236,1.048984,1.035903,0.973993,0.992473,1.015000,0.954313,0.956174,0.974032,0.990839,1.003359,0.966709,0.985554,1.003768,1.018555,0.999440,1.004774,0.980462,1.001820,1.013262,1.010000,0.991863,0.986632,0.975110,1.021525,1.007547,1.055420,1.008000,1.000000,1.050000,0.996551,1.000254,0.945276,1.018005,1.000000,1.042356,1.049552,1.040000,1.053541,1.041466,1.000000,1.038706,1.009515,1.016500,1.055850,1.010000,1.000000,1.023776,1.050000,0.993000,1.010000,0.992178,0.971140,0.965191,0.969095,0.976999,0.976227,1.020532,1.025125,1.015209,1.014590,1.000433,0.980890,0.974945,0.942873,0.972387,0.960470,1.000921,0.977728,0.958325,1.031028,1.012876,1.024438,1.012197,0.969485,1.050700,1.050700,1.032300,1.014500,1.050700,1.050700,1.050700,1.029000,1.050000,1.014500,1.050700,0.996700,1.021200,1.014500,1.001700,0.989300,1.050700,1.050700,1.014500,1.011774,0.994500,0.983335,0.976825,1.011711,1.002924,0.991387,1.002280,0.988722,0.964884,0.974704,0.970504,0.964756,0.965606,0.931742,0.944074,0.928799,0.997240,0.950422,0.959699,0.957027,0.939160,0.963555,0.950267,0.964683,0.979007,1.000000,0.978627,1.000000,1.000000,1.000000,0.975431,0.980460,0.979888,1.040517]   

    @test isapprox(vmag, vmag_matpower, rtol=1e-6)
end
