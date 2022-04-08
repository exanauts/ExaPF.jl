module PowerSystem

using Printf
using LinearAlgebra
using SparseArrays

import PowerModels

import Base: show, get

const PQ_BUS_TYPE = 1
const PV_BUS_TYPE = 2
const REF_BUS_TYPE  = 3

"""
    AbstractPowerSystem

First layer of the package. Store the topology of a given
transmission network, including:

- the power injection at each bus ;
- the admittance matrix ;
- the default voltage at each bus.

Data are imported either from a matpower file, or a PSSE file.

"""
abstract type AbstractPowerSystem end

"""
    AbstractNetworkElement

Abstraction for all physical elements being parts of a `AbstractPowerSystem`.
Elements are divided in

- transmission lines (`Lines`)
- buses (`Buses`)
- generators (`Generators`)

"""
abstract type AbstractNetworkElement end

"""
    Buses <: AbstractNetworkElement

Buses of a transmission network.
"""
struct Buses <: AbstractNetworkElement end

"""
    Lines <: AbstractNetworkElement

Lines of a transmission network.
"""
struct Lines <: AbstractNetworkElement end

"""
    Generators <: AbstractElement

Generators in a transmission network
"""
struct Generators <: AbstractNetworkElement end

"""
    AbstractNetworkAttribute

Attribute of a `AbstractPowerSystem`.

"""
abstract type AbstractNetworkAttribute end

"""
    NumberOfBuses <: AbstractNetworkAttribute

Number of buses in a `AbstractPowerSystem`.
"""
struct NumberOfBuses <: AbstractNetworkAttribute end

"""
    NumberOfLines <: AbstractNetworkAttribute

Number of lines in a `AbstractPowerSystem`.
"""
struct NumberOfLines <: AbstractNetworkAttribute end

"""
    NumberOfGenerators <: AbstractNetworkAttribute

Number of generators in a `AbstractPowerSystem`.
"""
struct NumberOfGenerators <: AbstractNetworkAttribute end

"""
    NumberOfPVBuses <: AbstractNetworkAttribute

Number of PV buses in a `AbstractPowerSystem`.
"""
struct NumberOfPVBuses <: AbstractNetworkAttribute end

"""
    NumberOfPQBuses <: AbstractNetworkAttribute

Number of PQ buses in a `AbstractPowerSystem`.
"""
struct NumberOfPQBuses <: AbstractNetworkAttribute end

"""
    NumberOfSlackBuses <: AbstractNetworkAttribute

Number of slack buses in a `AbstractPowerSystem`.
"""
struct NumberOfSlackBuses <: AbstractNetworkAttribute end

"""
    BaseMVA <: AbstractNetworkAttribute

Base MVA of the network.
"""
struct BaseMVA <: AbstractNetworkAttribute end

"""
    BusAdmittanceMatrix <: AbstractNetworkAttribute

Bus admittance matrix associated with the topology of the network.
"""
struct BusAdmittanceMatrix <: AbstractNetworkAttribute end

abstract type AbstractIndexing <: AbstractNetworkAttribute end

"""
    AllBusesIndex <: AbstractIndexing

Indexes of all the buses in a `AbstractPowerSystem`.
"""
struct AllBusesIndex <: AbstractIndexing end

"""
    PVIndexes <: AbstractIndexing

Indexes of the PV buses in a `AbstractPowerSystem`.
"""
struct PVIndexes <: AbstractIndexing end

"""
    PQIndexes <: AbstractIndexing

Indexes of the PQ buses in a `AbstractPowerSystem`.
"""
struct PQIndexes <: AbstractIndexing end

"""
    SlackIndexes <: AbstractIndexing

Indexes of the slack buses in a `AbstractPowerSystem`.
"""
struct SlackIndexes <: AbstractIndexing end

"""
    GeneratorIndexes <: AbstractIndexing

Indexes of the generators in a `AbstractPowerSystem`.
"""
struct GeneratorIndexes <: AbstractIndexing end

struct PVToGeneratorsIndex <: AbstractIndexing end
struct SlackToGeneratorsIndex <: AbstractIndexing end
struct AllGeneratorsIndex <: AbstractIndexing end

"""
    AbstractNetworkValues

Numerical values attached to the different attributes of the network.
"""
abstract type AbstractNetworkValues end

"""
    VoltageMagnitude <: AbstractNetworkValues

Magnitude `|v|` of the voltage `v = |v| exp(i θ)`.
"""
struct VoltageMagnitude <: AbstractNetworkValues end

"""
    VoltageAngle <: AbstractNetworkValues

Angle `θ` of the voltage `v = |v| exp(i θ)`.
"""
struct VoltageAngle <: AbstractNetworkValues end

"""
    ActivePower <: AbstractNetworkValues

Active power `P` of the complex power `S = P + iQ`.
"""
struct ActivePower <: AbstractNetworkValues end

"""
    ReactivePower <: AbstractNetworkValues

Reactive power `Q` of the complex power `S = P + iQ`.
"""
struct ReactivePower <: AbstractNetworkValues end

"""
    ActiveLoad <: AbstractNetworkValues

Active load `Pd` at buses.
"""
struct ActiveLoad <: AbstractNetworkValues end

"""
    ReactiveLoad <: AbstractNetworkValues

Reactive load `Qd` at buses.
"""
struct ReactiveLoad <: AbstractNetworkValues end

# Templating
"""
    get(pf::AbstractPowerSystem, attr::AbstractNetworkAttribute)

Return value of attribute `attr` in the `AbstractPowerSystem` object `pf`.

    get(pf::AbstractPowerSystem, attr::AbstractIndexing)

Return indexing corresponding to a subset of the buses.

## Examples

```julia
npq = get(pf, NumberOfPQBuses())
npv = get(pf, NumberOfPVBuses())

```
"""
function get end

"""
    bounds(pf::AbstractPowerSystem, attr::AbstractNetworkAttribute, val::AbstractNetworkValues)

Return lower and upper bounds corresponding to the admissible values
of the `AbstractNetworkAttribute` `attr`.

## Examples

```julia
p_min, p_max = bounds(pf, Generator(), ActivePower())
v_min, v_max = bounds(pf, Buses(), VoltageMagnitude())

```
"""
function bounds end

struct Branches{T}
    Yff::Vector{T}
    Yft::Vector{T}
    Ytf::Vector{T}
    Ytt::Vector{T}
    from_buses::Vector{Int}
    to_buses::Vector{Int}
end

# Utils
function get_bus_id_to_indexes(bus)
    BUS_I = IndexSet.idx_bus()[1]
    nbus = size(bus, 1)

    bus_id_to_indexes = Dict{Int,Int}()
    for i in 1:nbus
        busn = bus[i, BUS_I]
        bus_id_to_indexes[busn] = i
    end
    return bus_id_to_indexes
end

function get_bus_generators(buses, gens, bus_id_to_indexes)
    bus_gen = Dict{Int, Array{Int}}()
    GEN_BUS = IndexSet.idx_gen()[1]
    for g in 1:size(gens, 1)
        bus_id = gens[g, GEN_BUS]
        bus_index = bus_id_to_indexes[bus_id]
        if haskey(bus_gen, bus_index)
            push!(bus_gen[bus_index], g)
        else
            bus_gen[bus_index] = Int[g]
        end
    end
    return bus_gen
end

function get_active_branches(lines, remove_lines)
    if isempty(remove_lines)
        return lines
    else
        lines = lines[1:end .!= remove_lines, :]
        return lines
    end
end

include("indexes.jl")
using .IndexSet

include("utils.jl")
include("topology.jl")
include("power_network.jl")
include("matpower.jl")

end
