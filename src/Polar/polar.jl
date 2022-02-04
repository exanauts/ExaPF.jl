# Polar formulation

"""
    PolarForm{T, IT, VT, MT} <: AbstractFormulation

Wrap a [`PS.PowerNetwork`](@ref) network to load the data on
the target device (`CPU()` and `CUDADevice()` are currently supported).

"""
struct PolarForm{T, IT, VT, MT} <: AbstractFormulation where {T, IT, VT, MT}
    network::PS.PowerNetwork
    device::KA.Device
end

function PolarForm(pf::PS.PowerNetwork, device::KA.CPU)
    return PolarForm{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}(pf, device)
end
# Convenient constructor
PolarForm(datafile::String, device=CPU()) = PolarForm(PS.PowerNetwork(datafile), device)

# Getters (bridge to PowerNetwork)
get(polar::PolarForm, attr::PS.AbstractNetworkAttribute) = get(polar.network, attr)

number(polar::PolarForm, v::AbstractVariable) = length(mapping(polar, v))

# Default ordering in NetworkStack: [vmag, vang, pgen]

"""
    mapping(polar::PolarForm, ::State)

Return the mapping associated to the `State()` in [`NetworkStack`](@ref)
according to the polar formulation `PolarForm`.
"""
function mapping(polar::PolarForm, ::State)
    pf = polar.network
    nbus = get(polar, PS.NumberOfBuses())
    ref, pv, pq = pf.ref, pf.pv, pf.pq
    return Int[nbus .+ pv; nbus .+ pq; pq]
end

"""
    mapping(polar::PolarForm, ::Control)

Return the mapping associated to the `Control()` in [`NetworkStack`](@ref)
according to the polar formulation `PolarForm`.
"""
function mapping(polar::PolarForm, ::Control)
    pf = polar.network
    nbus = get(polar, PS.NumberOfBuses())
    ref, pv, pq = pf.ref, pf.pv, pf.pq
    genidx = Int[]
    for (idx, b) in enumerate(pf.gen2bus)
        if b != ref[1]
            push!(genidx, idx)
        end
    end
    return Int[ref; pv; 2*nbus .+ genidx]
end

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

