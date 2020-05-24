target = "cpu"
using PowerFlow
using PowerFlow.Parse
using PowerFlow.Network

include("pf.jl")
# datafile = "test/case14.raw"
datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_30R-025/scenario_1/case.raw"
@time sol, conv, res, avg_iter, gmres_iter = pf(datafile, 150)
println("avgerage gmres iterations: $avg_iter gmres iterations: $gmres_iter")
println("")