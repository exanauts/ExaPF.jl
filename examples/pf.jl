using ExaPF
using ExaPF.Parse
using ExaPF.PowerSystem
using KernelAbstractions

# file locations
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_30R-025/scenario_1/case.raw"
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
# datafile = "test/case14.raw"
# npartition: Number of partitions for the additive Schwarz preconditioner
function pf(datafile, npartition=2; solver="default", device = CPU())
    # read data
    println(solver)
    
    pf = PowerSystem.PowerNetwork(datafile)
    x = ExaPF.PowerSystem.get_x(pf)
    u = ExaPF.PowerSystem.get_u(pf)
    p = ExaPF.PowerSystem.get_p(pf)
    return solve(pf, x, u, p; npartitions=npartition,
                                solver=solver,
                                device=device)
end
