# Polar formulation

abstract type AbstractPolarFormulation{T, IT, VT, MT} <: AbstractFormulation end

# Getters (bridge to PowerNetwork)
get(polar::AbstractPolarFormulation, attr::PS.AbstractNetworkAttribute) = get(polar.network, attr)

number(polar::AbstractPolarFormulation, v::AbstractVariable) = length(mapping(polar, v))

"""
    PolarForm{T, IT, VT, MT} <: AbstractPolarFormulation

Implement the polar formulation associated to the network's equations.

Wrap a [`PS.PowerNetwork`](@ref) network to load the data on
the target device (`CPU()` and `CUDADevice()` are currently supported).

## Example
```jldoctest; setup=:(using ExaPF)
julia> const PS = ExaPF.PowerSystem;

julia> network_data = PS.load_case("case9.m");

julia> polar = PolarForm(network_data, ExaPF.CPU())
Polar formulation (instantiated on device CPU())
Network characteristics:
    #buses:      9  (#slack: 1  #PV: 2  #PQ: 6)
    #generators: 3
    #lines:      9
giving a mathematical formulation with:
    #controls:   5
    #states  :   14

```
"""
struct PolarForm{T, IT, VT, MT} <: AbstractPolarFormulation{T, IT, VT, MT}
    network::PS.PowerNetwork
    device::KA.Device
end

function PolarForm(pf::PS.PowerNetwork, device::KA.CPU)
    return PolarForm{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}(pf, device)
end
# Convenient constructor
PolarForm(datafile::String, device=CPU()) = PolarForm(PS.PowerNetwork(datafile), device)
PolarForm(polar::PolarForm, device=CPU()) = PolarForm(polar.network, device)

name(polar::PolarForm) = "Polar formulation"
nblocks(polar::PolarForm) = 1


"""
    BlockPolarForm{T, IT, VT, MT} <: AbstractFormulation

Block polar formulation: duplicates `k` different polar models
to evaluate them in parallel.

"""
struct BlockPolarForm{T, IT, VT, MT} <: AbstractPolarFormulation{T, IT, VT, MT}
    network::PS.PowerNetwork
    device::KA.Device
    k::Int
end
function BlockPolarForm(pf::PS.PowerNetwork, device, k::Int)
    return BlockPolarForm{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}(pf, device, k)
end
BlockPolarForm(datafile::String, k::Int, device=CPU()) = BlockPolarForm(PS.PowerNetwork(datafile), device, k)
BlockPolarForm(polar::PolarForm, k::Int) = BlockPolarForm(polar.network, polar.device, k)

name(polar::BlockPolarForm) = "$(polar.k)-BlockPolar formulation"
nblocks(polar::BlockPolarForm) = polar.k

function Base.show(io::IO, polar::AbstractPolarFormulation)
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
    print(io,   name(polar))
    println(io, " (instantiated on device $(polar.device))")
    println(io, "Network characteristics:")
    @printf(io, "    #buses:      %d  (#slack: %d  #PV: %d  #PQ: %d)\n", nbus, nref, npv, npq)
    println(io, "    #generators: ", ngen)
    println(io, "    #lines:      ", nlines)
    println(io, "giving a mathematical formulation with:")
    println(io, "    #controls:   ", n_controls)
    print(io,   "    #states  :   ", n_states)
end

include("utils.jl")
include("stacks.jl")
include("functions.jl")
include("first_order.jl")
include("second_order.jl")
include("newton.jl")
include("legacy.jl")

