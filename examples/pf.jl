using ExaPF
using ExaPF.Parse
using ExaPF.PowerSystem

# file locations
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_30R-025/scenario_1/case.raw"
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
# datafile = "test/case14.raw"
# npartition: Number of partitions for the additive Schwarz preconditioner
function pf(datafile, npartition=2, solver="default")
    # read data
    println(solver)
    
    pf = PowerSystem.PowerNetwork(datafile)
    return solve(pf, npartition, solver);
end
