target = "cpu"
using PowerFlow
using PowerFlow.Parse
using PowerFlow.Network
using Plots

include("pf.jl")
datafile = "test/case14.raw"
gmresblocks =    [2,4,8]
bicgstabblocks = [2,4,8]
outfile = "case14.txt"
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
# gmresblocks =    [8, 16, 32, 64, 128]
# bicgstabblocks = [8, 16, 32, 64, 128]
# outfile = "Network_13R-015.txt"
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_30R-025/scenario_1/case.raw"
# gmresblocks =    [8, 16, 32, 64, 128, 256]
# bicgstabblocks = [8, 16, 32, 64, 128, 256]
# outfile = "Network_30R-025.txt"
total_iter = []
average_iter = []
py = []
for blocks in gmresblocks
    @time sol, conv, res, avg_iter, gmres_iter = pf(datafile, blocks, "gmres")
    push!(total_iter, gmres_iter)
    push!(average_iter, avg_iter)
    println("avgerage gmres iterations: $avg_iter gmres iterations: $gmres_iter")
end
push!(py, total_iter)
push!(py, average_iter)

total_iter = []
average_iter = []
for blocks in bicgstabblocks
    @time sol, conv, res, avg_iter, gmres_iter = pf(datafile, blocks, "bicgstab")
    push!(total_iter, gmres_iter)
    push!(average_iter, avg_iter)
    println("avgerage bicgstab iterations: $avg_iter bicgstab iterations: $gmres_iter")
end
push!(py, total_iter)
push!(py, average_iter)
f = open(outfile, "w+")
@show py
println(f, py)
close(f)
f = open(outfile, "r+")
py = eval(Meta.parse(read(f, String)))
@show py
close(f)
# plot(bicgstabblocks, py, xlabel="Blocks", ylabel="Iterations", label = ["Total GMRES iter." "Avg. GMRES iter." "Total BICGSTAB iter." "Avg. BICGSTAB iter."])
