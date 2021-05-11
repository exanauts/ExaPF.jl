"""
    PowerNetwork <: AbstractPowerSystem

This structure contains constant parameters that define the topology and
physics of the power network.

The object `PowerNetwork` uses its own contiguous indexing for the buses.
The indexing is independent from those specified in the Matpower or the
PSSE input file. However, a correspondence between the two indexing
(Input indexing *to* `PowerNetwork` indexing) is stored inside the
attribute `bus_to_indexes`.

## Note
The object `PowerNetwork` is created in the host memory.
Use a `AbstractFormulation` object to move data to the target device.

"""
struct PowerNetwork <: AbstractPowerSystem
    vbus::Vector{Complex{Float64}}
    # Admittance matrix
    Ybus::SparseArrays.SparseMatrixCSC{Complex{Float64},Int64}
    # Lines
    lines::Branches{Complex{Float64}}
    # Data
    buses::Array{Float64, 2}
    branches::Array{Float64, 2}
    generators::Array{Float64, 2}
    costs::Union{Array{Float64, 2}, Nothing}
    baseMVA::Float64

    nbus::Int64
    ngen::Int64

    bustype::Vector{Int64}
    bus_to_indexes::Dict{Int, Int}
    ref::Vector{Int64}
    pv::Vector{Int64}
    pq::Vector{Int64}

    sbus::Vector{Complex{Float64}}
    sload::Vector{Complex{Float64}}

    function PowerNetwork(data::Dict{String, Array}; remove_lines=Int[], multi_generators=:aggregate)
        # Parsed data indexes
        BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
        LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()

        # retrive required data
        bus = data["bus"]
        gen = data["gen"]
        lines = data["branch"]
        SBASE = data["baseMVA"][1]
        cost_coefficients = Base.get(data, "cost", nothing)

        # BUSES
        bus_id_to_indexes = get_bus_id_to_indexes(bus)

        # GENERATORS
        if has_multiple_generators(gen) && multi_generators == :aggregate
            gen, σg = merge_multi_generators(gen)
            if !isnothing(cost_coefficients)
                cost_coefficients = merge_cost_coefficients(cost_coefficients, gen, σg)
            end
        end

        # LINES
        # Remove specified lines
        lines = get_active_branches(lines, remove_lines)

        # size of the system
        nbus = size(bus, 1)
        ngen = size(gen, 1)

        # COSTS
        if isnothing(cost_coefficients)
            @warn("[PS] Cost function not specified in dataset. Fallback to default coefficients.")
            # if not specified, costs is set by default to a # quadratic polynomial
            costs = zeros(ngen, 7)
            costs[:, 1] .= 2.0 # polynomial model
            costs[:, 2] .= 0.0 # no start-up cost
            costs[:, 3] .= 0.0 # no shutdown cost
            costs[:, 4] .= 3.0 # quadratic polynomial
            costs[:, 5] .= 0.0 # c₁
            costs[:, 6] .= 1.0 # c₂
            costs[:, 7] .= 0.0 # c₃
        else
            costs = cost_coefficients
        end
        # Check consistency of cost coefficients
        @assert size(costs, 1) == size(gen, 1)

        # obtain V0 from raw data
        vbus = zeros(Complex{Float64}, nbus)
        for i in 1:nbus
            vbus[i] = bus[i, VM]*exp(1im * pi/180 * bus[i, VA])
        end

        # form Y matrix
        topology = makeYbus(bus, lines, SBASE, bus_id_to_indexes)

        branches = Branches{Complex{Float64}}(
            topology.yff, topology.yft, topology.ytf, topology.ytt,
            topology.from_buses, topology.to_buses,
        )

        # bus type indexing
        ref, pv, pq, bustype = bustypeindex(bus, gen, bus_id_to_indexes)
        # check consistency
        ref_id = bus[ref, 1]
        @assert bus[ref, 2] == [REF_BUS_TYPE]
        if !(ref_id[1] in gen[:, 1])
            error("[PS] No generator attached to slack node.")
        end

        sbus, sload = assembleSbus(gen, bus, SBASE, bus_id_to_indexes)
        Ybus = topology.ybus

        new(vbus, Ybus, branches, bus, lines, gen, costs, SBASE, nbus, ngen, bustype, bus_id_to_indexes, ref, pv, pq, sbus, sload)
    end
end

function PowerNetwork(datafile::String; options...)
    data = import_dataset(datafile)
    return PowerNetwork(data; options...)
end

# Getters
## Network attributes
get(pf::PowerNetwork, ::NumberOfBuses) = pf.nbus
get(pf::PowerNetwork, ::NumberOfLines) = size(pf.branches, 1)
get(pf::PowerNetwork, ::NumberOfGenerators) = pf.ngen
get(pf::PowerNetwork, ::NumberOfPVBuses) = length(pf.pv)
get(pf::PowerNetwork, ::NumberOfPQBuses) = length(pf.pq)
get(pf::PowerNetwork, ::NumberOfSlackBuses) = length(pf.ref)

## Loads
get(pf::PowerNetwork, ::ActiveLoad) = real.(pf.sload)
get(pf::PowerNetwork, ::ReactiveLoad) = imag.(pf.sload)
function get(pf::PowerNetwork, ::ActivePower)
    GEN_BUS, PG, QG, _ = IndexSet.idx_gen()
    return pf.generators[:, PG] ./ pf.baseMVA
end
function get(pf::PowerNetwork, ::ReactivePower)
    GEN_BUS, PG, QG, _ = IndexSet.idx_gen()
    return pf.generators[:, QG] ./ pf.baseMVA
end

## Indexing
function get(pf::PowerNetwork, ::GeneratorIndexes)
    GEN_BUS = IndexSet.idx_gen()[1]
    gens = pf.generators
    ngens = size(gens, 1)
    # Create array on host memory
    indexing = zeros(Int, ngens)
    # Here, we keep the same ordering as specified in Matpower.
    for i in 1:ngens
        indexing[i] = pf.bus_to_indexes[gens[i, GEN_BUS]]
    end
    return indexing
end
get(pf::PowerNetwork, ::PVIndexes) = pf.pv
get(pf::PowerNetwork, ::PQIndexes) = pf.pq
get(pf::PowerNetwork, ::SlackIndexes) = pf.ref


# Pretty printing
function Base.show(io::IO, pf::PowerNetwork)
    println("Power Network characteristics:")
    @printf("\tBuses: %d. Slack: %d. PV: %d. PQ: %d\n", pf.nbus, length(pf.ref),
            length(pf.pv), length(pf.pq))
    println("\tGenerators: ", pf.ngen, ".")
    # Print system status
    @printf("\t==============================================\n")
    @printf("\tBUS \t TYPE \t VMAG \t VANG \t P \t Q\n")
    @printf("\t==============================================\n")

    for i=1:pf.nbus
        type = pf.bustype[i]
        vmag = abs(pf.vbus[i])
        vang = angle(pf.vbus[i])*(180.0/pi)
        pinj = real(pf.sbus[i])
        qinj = imag(pf.sbus[i])
        @printf("\t%d \t  %d \t %1.3f\t%3.2f\t%3.3f\t%3.3f\n", i,
                type, vmag, vang, pinj, qinj)
    end
end

# Some utils function

function bounds(pf::PowerNetwork, ::Buses, ::VoltageMagnitude)
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()

    bus = pf.buses
    v_min = convert.(Float64, bus[:, VMIN])
    v_max = convert.(Float64, bus[:, VMAX])
    return v_min, v_max
end

function bounds(pf::PowerNetwork, ::Generators, ::ActivePower)
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()

    gens = pf.generators
    baseMVA = pf.baseMVA

    p_min = convert.(Float64, gens[:, PMIN] / baseMVA)
    p_max = convert.(Float64, gens[:, PMAX] / baseMVA)
    return p_min, p_max
end

function bounds(pf::PowerNetwork, ::Generators, ::ReactivePower)
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()

    gens = pf.generators
    baseMVA = pf.baseMVA

    q_min = convert.(Float64, gens[:, QMIN] / baseMVA)
    q_max = convert.(Float64, gens[:, QMAX] / baseMVA)
    return q_min, q_max
end

function bounds(pf::PowerNetwork, ::Lines, ::ActivePower)
    RATE_A = IndexSet.idx_branch()[6]
    n_lines = get(pf, NumberOfLines())
    # Flow min is always set equal to 0
    flow_min = zeros(n_lines)
    flow_max = (pf.branches[:, RATE_A] ./ pf.baseMVA).^2
    # According to the spec, if RATE_A is equal to 0, then the flow
    # is unconstrained.
    unlimited = findall(isequal(0), flow_max)
    flow_max[unlimited] .= Inf
    return (flow_min, flow_max)
end

"""
    get_costs_coefficients(pf::PowerNetwork)

Return coefficients for costs function.

"""
function get_costs_coefficients(pf::PowerNetwork)
    # indexes
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()
    MODEL, STARTUP, SHUTDOWN, NCOST, COST = IndexSet.idx_cost()

    ref = pf.ref
    pv = pf.pv
    pq = pf.pq
    b2i = pf.bus_to_indexes

    # Matpower assumes gens are ordered. Generator in row i has its cost on row i
    # of the cost table.
    gens = pf.generators
    baseMVA = pf.baseMVA
    bus = pf.buses
    ngens = size(gens)[1]
    nbus = size(bus)[1]

    # initialize cost
    # store coefficients in a Float64 array, with 4 columns:
    # - 1st column: bus type
    # - 2nd column: constant coefficient c0
    # - 3rd column: coefficient c1
    # - 4th column: coefficient c2
    coefficients = zeros(ngens, 4)

    cost_data = pf.costs
    # iterate generators and check if pv or ref.
    for i = 1:ngens
        # only 2nd degree polynomial implemented for now.
        @assert cost_data[i, MODEL] == 2
        genbus = b2i[gens[i, GEN_BUS]]
        bustype = bus[genbus, BUS_TYPE]

        # polynomial coefficients
        if cost_data[i, NCOST] == 3       # quadratic model
            c0 = cost_data[i, COST+2]
            c1 = cost_data[i, COST+1] * baseMVA
            c2 = cost_data[i, COST] * baseMVA^2
        elseif cost_data[i, NCOST] == 2   # linear model
            c0 = cost_data[i, COST+1]
            c1 = cost_data[i, COST] * baseMVA
            c2 = 0.0
        end

        coefficients[i, :] .= (bustype, c0, c1, c2)
    end
    return coefficients
end

