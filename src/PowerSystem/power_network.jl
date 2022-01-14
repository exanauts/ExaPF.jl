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
    # Generators From/To Buses Indexes
    gen2bus::Vector{Int64}
    ref2gen::Vector{Int64}
    pv2gen::Vector{Int64}

    sbus::Vector{Complex{Float64}}
    sload::Vector{Complex{Float64}}

    function PowerNetwork(data::Dict{String, Array}; remove_lines=Int[])
        # Parsed data indexes
        BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
        LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()

        # retrive required data
        bus = data["bus"]::Array{Float64, 2}
        gen = data["gen"]::Array{Float64, 2}
        lines = data["branch"]::Array{Float64, 2}
        SBASE = data["baseMVA"][1]::Float64
        cost_coefficients = Base.get(data, "cost", nothing)

        # BUSES
        bus_id_to_indexes = get_bus_id_to_indexes(bus)

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
        ref, pv, pq, bustype, inactive_generators = bustypeindex(bus, gen, bus_id_to_indexes)
        # check consistency
        ref_id = bus[ref, 1]
        @assert bus[ref, 2] == [REF_BUS_TYPE]
        if !(ref_id[1] in gen[:, 1])
            error("[PS] No generator attached to slack node.")
        end
        if !isempty(inactive_generators)
            println("[PS] Found $(length(inactive_generators)) inactive generators.")
        end

        sbus, sload = assembleSbus(gen, bus, SBASE, bus_id_to_indexes)
        gen2bus = generators_to_buses(gen, bus_id_to_indexes)
        pv2gen, ref2gen = buses_to_generators(gen2bus, pv, ref)
        Ybus = topology.ybus

        new(vbus, Ybus, branches, bus, lines, gen, costs, SBASE, nbus, ngen, bustype, bus_id_to_indexes,
            ref, pv, pq, gen2bus, ref2gen, pv2gen, sbus, sload)
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
get(pf::PowerNetwork, ::GeneratorIndexes) = pf.gen2bus
get(pf::PowerNetwork, ::PVIndexes) = pf.pv
get(pf::PowerNetwork, ::PQIndexes) = pf.pq
get(pf::PowerNetwork, ::SlackIndexes) = pf.ref
get(pf::PowerNetwork, ::AllBusesIndex) = (pf.ref, pf.pv, pf.pq)
get(pf::PowerNetwork, ::SlackToGeneratorsIndex) = pf.ref2gen
get(pf::PowerNetwork, ::PVToGeneratorsIndex) = pf.pv2gen
get(pf::PowerNetwork, ::AllGeneratorsIndex) = (pf.gen2bus, pf.ref2gen, pf.pv2gen)

has_inactive_generators(pf::PowerNetwork) = any(isequal(0), view(pf.generators, :, 8))
active_generators(pf::PowerNetwork) = findall(isequal(1), view(pf.generators, :, 8))
inactive_generators(pf::PowerNetwork) = findall(isequal(0), view(pf.generators, :, 8))

# Pretty printing
function Base.show(io::IO, pf::PowerNetwork)
    println(io, "PowerNetwork object with:")
    @printf(io, "    Buses: %d (Slack: %d. PV: %d. PQ: %d)\n", pf.nbus, length(pf.ref),
            length(pf.pv), length(pf.pq))
    println(io, "    Generators: ", pf.ngen, ".")
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
    if has_inactive_generators(pf)
        inactive_gens = inactive_generators(pf)
        # Set lower and upper bounds to 0 for inactive generators
        p_min[inactive_gens] .= 0.0
        p_max[inactive_gens] .= 0.0
    end
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
    if has_inactive_generators(pf)
        inactive_gens = inactive_generators(pf)
        # Set lower and upper bounds to 0 for inactive generators
        q_min[inactive_gens] .= 0.0
        q_max[inactive_gens] .= 0.0
    end
    return q_min, q_max
end

function bounds(pf::PowerNetwork, ::Lines, ::ActivePower)
    RATE_A = IndexSet.idx_branch()[6]
    n_lines = get(pf, NumberOfLines())
    # Flow min is not bounded below
    flow_min = fill(-Inf, n_lines)
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
    ngens = size(gens, 1)
    nbus = size(bus, 1)

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

        coefficients[i, 1] = bustype
        coefficients[i, 2] = c0
        coefficients[i, 3] = c1
        coefficients[i, 4] = c2
    end
    return coefficients
end

function get_basis_matrix(pf::PowerNetwork)
    nb = pf.nbus
    nl = size(pf.branches, 1)
    Yff, Yft, Ytf, Ytt = pf.lines.Yff, pf.lines.Yft, pf.lines.Ytf, pf.lines.Ytt
    f, t = pf.lines.from_buses, pf.lines.to_buses

    Cf = sparse(f, 1:nl, ones(nl), nb, nl)       # connection matrix for line & from buses
    Ct = sparse(t, 1:nl, ones(nl), nb, nl)       # connection matrix for line & to buses

    ysh = (pf.buses[:, 5] .+ 1im .* pf.buses[:, 6]) ./ pf.baseMVA # vector of shunt admittances
    Ysh = sparse(1:nb, 1:nb, ysh, nb, nb)

    # Build matrix
    Yc = Cf * Diagonal(Yft) + Ct * Diagonal(Ytf)
    Ys = Cf * Diagonal(Yft) - Ct * Diagonal(Ytf)
    Yd = Cf * Diagonal(Yff) * Cf' + Ct * Diagonal(Ytt) * Ct' + Ysh

    return [-real(Yc) -imag(Ys) -real(Yd);
             imag(Yc)  -real(Ys)  imag(Yd)]
end

function get_line_flow_matrices(pf::PowerNetwork)
    nb = pf.nbus
    nl = size(pf.branches, 1)
    Yff, Yft, Ytf, Ytt = pf.lines.Yff, pf.lines.Yft, pf.lines.Ytf, pf.lines.Ytt

    yff = Diagonal(Yff)
    yft = Diagonal(Yft)
    ytf = Diagonal(Ytf)
    ytt = Diagonal(Ytt)

    f, t = pf.lines.from_buses, pf.lines.to_buses

    Cf = sparse(f, 1:nl, ones(nl), nb, nl)       # connection matrix for line & from buses
    Ct = sparse(t, 1:nl, ones(nl), nb, nl)       # connection matrix for line & to buses

    # Build matrix
    Lfp = [real(yft)  imag(yft)  real(yff) * Cf']
    Lfq = [-imag(yft) real(yft) -imag(yff) * Cf']
    Ltp = [real(ytf)  -imag(ytf)  real(ytt) * Ct']
    Ltq = [-imag(ytf) -real(ytf) -imag(ytt) * Ct']
    return (Lfp, Lfq, Ltp, Ltq)
end

