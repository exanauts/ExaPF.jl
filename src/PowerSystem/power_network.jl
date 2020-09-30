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
The object `PowerNetwork` is created in the main memory.
Use a `AbstractFormulation` object to move data to the target device.

"""
struct PowerNetwork <: AbstractPowerSystem
    vbus::Vector{Complex{Float64}}
    Ybus::SparseArrays.SparseMatrixCSC{Complex{Float64},Int64}
    data::Dict{String,Array}

    nbus::Int64
    ngen::Int64

    bustype::Vector{Int64}
    bus_to_indexes::Dict{Int, Int}
    ref::Vector{Int64}
    pv::Vector{Int64}
    pq::Vector{Int64}

    sbus::Vector{Complex{Float64}}
    sload::Vector{Complex{Float64}}

    function PowerNetwork(datafile::String, data_format::Int64=0)

        if data_format == 0
            println("Reading PSSE format")
            data_raw = ParsePSSE.parse_raw(datafile)
            data, bus_id_to_indexes = ParsePSSE.raw_to_exapf(data_raw)
        elseif data_format == 1
            data_mat = ParseMAT.parse_mat(datafile)
            data, bus_id_to_indexes = ParseMAT.mat_to_exapf(data_mat)
        end
        # Parsed data indexes
        BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
        LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()

        # retrive required data
        bus = data["bus"]
        gen = data["gen"]
        SBASE = data["baseMVA"][1]

        # size of the system
        nbus = size(bus, 1)
        ngen = size(gen, 1)

        # obtain V0 from raw data
        vbus = zeros(Complex{Float64}, nbus)
        for i in 1:nbus
            vbus[i] = bus[i, VM]*exp(1im * pi/180 * bus[i, VA])
        end

        # form Y matrix
        Ybus = makeYbus(data, bus_id_to_indexes)

        # bus type indexing
        ref, pv, pq, bustype = bustypeindex(bus, gen, bus_id_to_indexes)

        sbus, sload = assembleSbus(gen, bus, SBASE, bus_id_to_indexes)

        new(vbus, Ybus, data, nbus, ngen, bustype, bus_id_to_indexes, ref, pv, pq, sbus, sload)
    end
end

# Getters
## Network attributes
get(pf::PowerNetwork, ::NumberOfBuses) = pf.nbus
get(pf::PowerNetwork, ::NumberOfLines) = size(pf.data["branch"], 1)
get(pf::PowerNetwork, ::NumberOfGenerators) = pf.ngen
get(pf::PowerNetwork, ::NumberOfPVBuses) = length(pf.pv)
get(pf::PowerNetwork, ::NumberOfPQBuses) = length(pf.pq)
get(pf::PowerNetwork, ::NumberOfSlackBuses) = length(pf.ref)

## Indexing
function get(pf::PowerNetwork, ::GeneratorIndexes)
    GEN_BUS = IndexSet.idx_gen()[1]
    gens = pf.data["gen"]
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

    bus = pf.data["bus"]
    v_min = convert.(Float64, bus[:, VMIN])
    v_max = convert.(Float64, bus[:, VMAX])
    return v_min, v_max
end

function bounds(pf::PowerNetwork, ::Generator, ::ActivePower)
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()

    gens = pf.data["gen"]
    baseMVA = pf.data["baseMVA"][1]

    p_min = convert.(Float64, gens[:, PMIN] / baseMVA)
    p_max = convert.(Float64, gens[:, PMAX] / baseMVA)
    return p_min, p_max
end

function bounds(pf::PowerNetwork, ::Generator, ::ReactivePower)
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()

    gens = pf.data["gen"]
    baseMVA = pf.data["baseMVA"][1]

    q_min = convert.(Float64, gens[:, QMIN] / baseMVA)
    q_max = convert.(Float64, gens[:, QMAX] / baseMVA)
    return q_min, q_max
end

"""
    get_costs_coefficients(pf::PowerNetwork)

Return coefficients for costs function

TODO: how to deal with piecewise polynomial function?
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
    gens = pf.data["gen"]
    baseMVA = pf.data["baseMVA"][1]
    bus = pf.data["bus"]
    ngens = size(gens)[1]
    nbus = size(bus)[1]

    # initialize cost
    # store coefficients in a Float64 array, with 4 columns:
    # - 1st column: bus type
    # - 2nd column: constant coefficient c0
    # - 3rd column: coefficient c1
    # - 4th column: coefficient c2
    coefficients = zeros(ngens, 4)
    # If cost is not specified, we return the array coefficients as is
    if !haskey(pf.data, "cost")
        @warn("PowerSystem: cost is not specified in PowerNetwork dataset")
        return coefficients
    end

    cost_data = pf.data["cost"]
    # iterate generators and check if pv or ref.
    for i = 1:ngens
        # only 2nd degree polynomial implemented for now.
        @assert cost_data[i, MODEL] == 2
        @assert cost_data[i, NCOST] == 3
        genbus = b2i[gens[i, GEN_BUS]]
        bustype = bus[genbus, BUS_TYPE]

        # polynomial coefficients
        # TODO: currently scale by baseMVA. Is it a good idea?
        c0 = cost_data[i, COST][3]
        c1 = cost_data[i, COST][2] * baseMVA
        c2 = cost_data[i, COST][1] * baseMVA^2
        coefficients[i, :] .= (bustype, c0, c1, c2)
    end
    return coefficients
end

