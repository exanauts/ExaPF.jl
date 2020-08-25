using ExaPF
using KernelAbstractions

import ExaPF: ParsePSSE, PowerSystem, IndexSet

# file locations
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_30R-025/scenario_1/case.raw"
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
# datafile = "test/case14.raw"
# npartition: Number of partitions for the additive Schwarz preconditioner
function pf(datafile, npartition=2; solver="default", device = CPU())
    # Create a powersystem object:
    pf = ExaPF.PowerSystem.PowerNetwork(datafile, 1)

    # Retrieve initial state of network
    pbus = real.(pf.sbus)
    qbus = imag.(pf.sbus)
    vmag = abs.(pf.vbus)
    vang = angle.(pf.vbus)
    
    x = ExaPF.PowerSystem.get_x(pf, vmag, vang, pbus, qbus)
    u = ExaPF.PowerSystem.get_u(pf, vmag, vang, pbus, qbus)
    p = ExaPF.PowerSystem.get_p(pf, vmag, vang, pbus, qbus)
    return solve(pf, x, u, p; npartitions=npartition,
                                solver=solver,
                                device=device)
end
