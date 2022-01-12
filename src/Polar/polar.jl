# Polar formulation

"""
    PolarForm{T, IT, VT, MT}

Wrap a [`PS.PowerNetwork`](@ref) network to move the data on
the target device (`CPU()` and `CUDADevice()` are currently supported).

"""
struct PolarForm{T, IT, VT, MT} <: AbstractFormulation where {T, IT, VT, MT}
    network::PS.PowerNetwork
    device::KA.Device
end

function PolarForm(pf::PS.PowerNetwork, device::KA.CPU)
    return PolarForm{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}(pf, device)
end
function PolarForm(pf::PS.PowerNetwork, device::KA.GPU)
    return PolarForm{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}(pf, device)
end

# Convenient constructor
PolarForm(datafile::String, device) = PolarForm(PS.PowerNetwork(datafile), device)

# Default ordering: [vmag, vang, pgen]
function my_map(polar::PolarForm, ::State)
    pf = polar.network
    nbus = get(polar, PS.NumberOfBuses())
    ref, pv, pq = pf.ref, pf.pv, pf.pq
    return Int[nbus .+ pv; nbus .+ pq; pq]
end
function my_map(polar::PolarForm, ::Control)
    pf = polar.network
    nbus = get(polar, PS.NumberOfBuses())
    ref, pv, pq = pf.ref, pf.pv, pf.pq
    pv2gen = polar.network.pv2gen
    return Int[ref; pv; 2*nbus .+ pv2gen]
end

number(polar::PolarForm, v::AbstractVariable) = length(my_map(polar, v))

# Getters
get(polar::PolarForm, attr::PS.AbstractNetworkAttribute) = get(polar.network, attr)

function Base.show(io::IO, polar::PolarForm)
    # Network characteristics
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    nlines = PS.get(polar.network, PS.NumberOfLines())
    # Polar formulation characteristics
    n_states = 2*npq + npv
    n_controls = nref + npv + ngen - 1
    print(io,   "Polar formulation model")
    println(io, " (instantiated on device $(polar.device))")
    println(io, "Network characteristics:")
    @printf(io, "    #buses:      %d  (#slack: %d  #PV: %d  #PQ: %d)\n", nbus, nref, npv, npq)
    println(io, "    #generators: ", ngen)
    println(io, "    #lines:      ", nlines)
    println(io, "giving a mathematical formulation with:")
    println(io, "    #controls:   ", n_controls)
    print(io,   "    #states  :   ", n_states)
end

include("functions.jl")
include("first_order.jl")
include("second_order.jl")
include("newton.jl")
include("legacy.jl")

