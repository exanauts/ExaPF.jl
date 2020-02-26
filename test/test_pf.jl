using Test

# file locations
# raw_data = "GO-Data/datasets/Trial_3_Real-Time/Network_30R-025/scenario_1/case.raw"
# raw_data = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
raw_data = "test/case14.raw"
# Set this to "cpu" or "cuda" 
target = "cpu"


data_parser ="../src/parse.jl"
raw_parser ="../src/parse_raw.jl"
network ="../src/network.jl"
pflow ="../src/powerflow.jl"

# imports
include(data_parser)
include(raw_parser)
include(network)
include(pflow)
using .PowerFlow
using .Network

# read data
data = parse_raw(raw_data)

BUS_B, BUS_AREA, BUS_VM, BUS_VA, BUS_NVHI, BUS_NVLO, BUS_EVHI,
  BUS_EVLO, BUS_TYPE = idx_bus()

bus = data["BUS"]
nbus = size(bus, 1)

# obtain V0 from raw data
V = Array{Complex{Float64}}(undef, nbus)

for i in 1:nbus
  V[i] = bus[i, BUS_VM]*exp(1im * pi/180 * bus[i, BUS_VA])
end

# form Y matrix
Ybus, Yf_br, Yt_br, Yf_tr, Yt_tr = Network.makeYbus(data);

vsol, conv, norm = PowerFlow.newtonpf(V, Ybus, data);

# test convergence is OK
@test conv
# test norm is minimized
@test norm < 1e-7
