# Verify solutions against matpower results
using Test
using ExaPF

import ExaPF: ParseMAT, PowerSystem, IndexSet

@testset "Power flow 9 bus case" begin
    datafile = "test/case9.m"
    pf = PowerSystem.PowerNetwork(datafile, 1)
    x = ExaPF.PowerSystem.get_x(pf)
    u = ExaPF.PowerSystem.get_u(pf)
    p = ExaPF.PowerSystem.get_p(pf)

    ExaPF.solve(pf, x, u, p)

    # test impedance matrix entries
    @test isapprox(real(pf.Ybus[1, 1]), 0.0)
    @test isapprox(imag(pf.Ybus[1, 1]), -17.3611111)

	S = [0.0000 + 0.0000im, 1.6300 + 0.0000im, 0.8500 + 0.0000im, 0.0000 + 0.0000im,
		-0.9000 - 0.3000im, 0.0000 + 0.0000im, -1.0000 - 0.3500im, 0.0000 + 0.0000im,
		-1.2500 - 0.5000im]

    @test isapprox(S, pf.Sbus)


    c = ExaPF.cost_function(pf, x, u, p)
    dCdx, dCdu = ExaPF.cost_gradients(pf, x, u, p)
end
