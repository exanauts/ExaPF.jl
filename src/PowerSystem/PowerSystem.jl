module PowerSystem

using Printf
using SparseArrays
using ..ExaPF: ParsePSSE, ParseMAT, IndexSet, Spmat

import Base: show

const PQ_BUS_TYPE = 1
const PV_BUS_TYPE = 2
const REF_BUS_TYPE  = 3

"""
    AbstractPowerSystem

First layer of the package. Store the topology of a given
transmission network, including:

- the power injection at each bus
- the admittance matrix
- the default voltage at each bus

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
    Generator <: AbstractElement

Generator in a transmission network
"""
struct Generator <: AbstractNetworkElement end

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


abstract type AbstractIndexing <: AbstractNetworkAttribute end

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


# TODO: replace const *_BUS_TYPE with this enum
@enum BusType begin
    PQ=1
    PV=2
    Slack=3
end

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
index_pv = get(pf, PVIndexes()())
index_gen = get(pf, GeneratorIndexes()())

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

include("topology.jl")
include("power_network.jl")

end
