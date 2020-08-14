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

    # test impedance matrix entries

    # solve power flow
    ExaPF.solve(pf, x, u, p)

    c = ExaPF.cost_function(pf, x, u, p)
    dCdx, dCdu = ExaPF.cost_gradients(pf, x, u, p)
end
