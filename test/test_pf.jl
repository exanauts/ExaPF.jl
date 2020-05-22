using PowerFlow
using PowerFlow.Parse
using PowerFlow.Network

# file locations
# raw_data = "GO-Data/datasets/Trial_3_Real-Time/Network_30R-025/scenario_1/case.raw"
# raw_data = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
raw_data = "test/case14.raw"

# read data
data = Parse.parse_raw(raw_data)

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

vsol, conv, res = PowerFlow.newtonpf(pf);