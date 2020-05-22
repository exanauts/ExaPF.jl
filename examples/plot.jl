target = "cpu"
using PowerFlow
using PowerFlow.Parse
using PowerFlow.Network

include("pf.jl")
# datafile = "test/case14.raw"
datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
@time sol, conv, res = pf(datafile, 10)
println("")
@time sol, conv, res = pf(datafile, 10)
println("")