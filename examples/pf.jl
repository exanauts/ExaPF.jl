using ExaPF
using ExaPF.Parse
using ExaPF.Network

# file locations
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_30R-025/scenario_1/case.raw"
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
# datafile = "test/case14.raw"
# npartition: Number of partitions for the additive Schwarz preconditioner
function pf(datafile, npartition=2, solver="default")
# read data
data = Parse.parse_raw(datafile)
println(solver)

BUS_B, BUS_AREA, BUS_VM, BUS_VA, BUS_NVHI, BUS_NVLO, BUS_EVHI,
  BUS_EVLO, BUS_TYPE = Parse.idx_bus()

bus = data["BUS"]
nbus = size(bus, 1)

# obtain V0 from raw data
V = Array{Complex{Float64}}(undef, nbus)

for i in 1:nbus
  V[i] = bus[i, BUS_VM]*exp(1im * pi/180 * bus[i, BUS_VA])
end

# form Y matrix
Ybus, Yf_br, Yt_br, Yf_tr, Yt_tr = Network.makeYbus(data);

# V, Ybus, data
pf = Pf(V, Ybus, data)

return solve(pf, npartition, solver);
end
