target = "cpu"
using ExaPF
using ExaPF.Network
using DelimitedFiles

include("pf.jl")
datafile = "test/case14.raw"
gmresblocks =    [2,4,8]
bicgstabblocks = [2,4,8]
outfile = "./plots/case14.txt"
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
# gmresblocks =    [8, 16, 32, 64, 128]
# bicgstabblocks = [8, 16, 32, 64, 128]
# outfile = "./plots/Network_13R-015.txt"
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_30R-025/scenario_1/case.raw"
# gmresblocks =    [16, 32, 64, 128, 256]
# bicgstabblocks = [16, 32, 64, 128, 256]
# outfile = "./plots/Network_30R-025.txt"
linsol_iters = []
first_iters = []
py = []
push!(py, gmresblocks)
for blocks in gmresblocks
  @time sol, conv, res, first_iter, linsol_iter = pf(datafile, blocks, "gmres")
  push!(linsol_iters, linsol_iter)
  push!(first_iters, first_iter)
  println("first gmres iteration: $first_iter, total gmres iterations: $linsol_iter")
end
push!(py, linsol_iters)
push!(py, first_iters)

linsol_iters = []
first_iters = []
for blocks in bicgstabblocks
  @time sol, conv, res, first_iter, linsol_iter = pf(datafile, blocks, "bicgstab")
  push!(linsol_iters, linsol_iter)
  push!(first_iters, first_iter)
  println("first bicgstab iteration: $first_iter, total bicgstab iterations: $linsol_iter")
end
push!(py, linsol_iters)
push!(py, first_iters)
f = open(outfile, "w+")
writedlm(f, py,',')
close(f)
