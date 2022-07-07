# Polar formulation

"""
    PolarForm{T, IT, VT, MT} <: AbstractFormulation

Wrap a [`PS.PowerNetwork`](@ref) network to load the data on
the target device (`CPU()` and `CUDADevice()` are currently supported).

## Example
```jldoctest; setup=:(using ExaPF)
julia> const PS = ExaPF.PowerSystem;

julia> network_data = PS.load_case("case9.m");

julia> polar = PolarForm(network_data, ExaPF.CPU())
Polar formulation model (instantiated on device CPU())
Network characteristics:
    #buses:      9  (#slack: 1  #PV: 2  #PQ: 6)
    #generators: 3
    #lines:      9
giving a mathematical formulation with:
    #controls:   5
    #states  :   14

```
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
PolarForm(polar::PolarForm, device=CPU()) = PolarForm(polar.network, device)

"""
    load_polar(case, device=CPU(); dir=PS.EXADATA)

Load a [`PolarForm`](@ref) instance from the specified
benchmark library `dir` on the target `device` (default is `CPU`).
ExaPF uses two different benchmark libraries: MATPOWER (`dir=EXADATA`)
and PGLIB-OPF (`dir=PGLIB`).

## Examples
```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9")
Polar formulation model (instantiated on device CPU())
Network characteristics:
    #buses:      9  (#slack: 1  #PV: 2  #PQ: 6)
    #generators: 3
    #lines:      9
giving a mathematical formulation with:
    #controls:   5
    #states  :   14

```

"""
function load_polar(case, device=CPU(); dir=PS.EXADATA)
    return PolarForm(PS.load_case(case, dir), device)
end

# Getters (bridge to PowerNetwork)
get(polar::PolarForm, attr::PS.AbstractNetworkAttribute) = get(polar.network, attr)

number(polar::PolarForm, v::AbstractVariable) = length(mapping(polar, v))


# Default ordering in NetworkStack: [vmag, vang, pgen]

"""
    mapping(polar::PolarForm, ::Control)

Return the mapping associated to the `Control()` in [`NetworkStack`](@ref)
according to the polar formulation `PolarForm`.

## Examples
```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9");

julia> mapu = ExaPF.mapping(polar, Control())
5-element Vector{Int64}:
  1
  2
  3
 20
 21

```

"""
function mapping(polar::PolarForm, ::Control, k::Int=1)
    pf = polar.network
    nbus = get(polar, PS.NumberOfBuses())
    ngen = polar.network.ngen
    ref, pv = pf.ref, pf.pv
    genidx = Int[]
    for (idx, b) in enumerate(pf.gen2bus)
        if b != ref[1]
            push!(genidx, idx)
        end
    end
    nu = (length(pv) + length(ref) + length(genidx)) * k
    mapu = zeros(Int, nu)

    shift_mag = 0
    shift_pgen = 2 * nbus * k
    index = 1
    for i in 1:k
        for j in ref
            mapu[index] = j + (i-1) * nbus + shift_mag
            index += 1
        end
        for j in pv
            mapu[index] = j + (i-1) * nbus + shift_mag
            index += 1
        end
        for j in genidx
            mapu[index] = j + (i-1) * ngen + shift_pgen
            index += 1
        end
    end
    return mapu
end

"""
    mapping(polar::PolarForm, ::State)

Return the mapping associated to the `State()` in [`NetworkStack`](@ref)
according to the polar formulation `PolarForm`.

## Examples
```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9");

julia> mapu = ExaPF.mapping(polar, State())
14-element Vector{Int64}:
 11
 12
 13
 14
 15
 16
 17
 18
  4
  5
  6
  7
  8
  9

```

"""
function mapping(polar::PolarForm, ::State, k::Int=1)
    pf = polar.network
    nbus = get(polar, PS.NumberOfBuses())
    ref, pv, pq = pf.ref, pf.pv, pf.pq
    nx = (length(pv) + 2*length(pq)) * k
    mapx = zeros(Int, nx)

    shift_ang = k * nbus
    shift_mag = 0
    index = 1
    for i in 1:k
        for j in [pv; pq]
            mapx[index] = j + (i-1) * nbus + shift_ang
            index += 1
        end
        for j in pq
            mapx[index] = j + (i-1) * nbus + shift_mag
            index += 1
        end
    end
    return mapx
end

function mapping(polar::PolarForm, ::AllVariables, k::Int=1)
    pf = polar.network
    nbus = get(polar, PS.NumberOfBuses())
    ngen = polar.network.ngen
    ref, pv, pq = pf.ref, pf.pv, pf.pq
    genidx = Int[]
    for (idx, b) in enumerate(pf.gen2bus)
        if b != ref[1]
            push!(genidx, idx)
        end
    end
    nx = (length(pv) + 2*length(pq)) * k
    nu = (length(pv) + length(ref) + length(genidx)) * k
    mapxu = zeros(Int, nx+nu)

    shift_mag = 0
    shift_ang = k * nbus
    shift_pgen = 2 * nbus * k
    index = 1
    for i in 1:k
        # / x
        for j in [pv; pq]
            mapxu[index] = j + (i-1) * nbus + shift_ang
            index += 1
        end
        for j in pq
            mapxu[index] = j + (i-1) * nbus + shift_mag
            index += 1
        end
        # / u
        for j in [ref; pv]
            mapxu[index] = j + (i-1) * nbus + shift_mag
            index += 1
        end
        for j in genidx
            mapxu[index] = j + (i-1) * ngen + shift_pgen
            index += 1
        end
    end
    return mapxu
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

include("stacks.jl")
include("functions.jl")
include("first_order.jl")
include("second_order.jl")
include("newton.jl")
include("legacy.jl")

