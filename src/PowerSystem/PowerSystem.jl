module PowerSystem

using Printf
using SparseArrays
using ..ExaPF: ParsePSSE, ParseMAT, IndexSet, Spmat

import Base: show

const PQ_BUS_TYPE = 1
const PV_BUS_TYPE = 2
const REF_BUS_TYPE  = 3

abstract type AbstractNetworkAttribute end
struct NumberOfBuses <: AbstractNetworkAttribute end
struct NumberOfLines <: AbstractNetworkAttribute end
struct NumberOfGenerators <: AbstractNetworkAttribute end
struct NumberOfPVBuses <: AbstractNetworkAttribute end
struct NumberOfPQBuses <: AbstractNetworkAttribute end
struct NumberOfSlackBuses <: AbstractNetworkAttribute end

abstract type AbstractBuses end
struct Buses <: AbstractBuses end

abstract type AbstractBusType end
struct PVBuses <: AbstractBusType end
struct PQBuses <: AbstractBusType end
struct SlackBuses <: AbstractBusType end

abstract type AbstractIndexing end
struct PVIndexes <: AbstractIndexing end
struct PQIndexes <: AbstractIndexing end
struct SlackIndexes <: AbstractIndexing end
struct GeneratorIndexes <: AbstractIndexing end
struct PVGeneratorIndexes <: AbstractIndexing end
struct SlackGeneratorIndexes <: AbstractIndexing end

# TODO: replace const *_BUS_TYPE with this enum
@enum BusType begin
    PQ=1
    PV=2
    Slack=3
end

abstract type AbstractLines end
struct Lines <: AbstractLines end

abstract type AbstractGenerator end
struct Generator <: AbstractGenerator end

abstract type AbstractValues end
struct VoltageMagnitude <: AbstractValues end
struct VoltageAngle <: AbstractValues end
struct ActivePower <: AbstractValues end
struct ReactivePower <: AbstractValues end

abstract type AbstractPowerSystem end
# Templating
function get(::AbstractPowerSystem, ::AbstractNetworkAttribute) end

include("network_topology.jl")
include("power_network.jl")

end
