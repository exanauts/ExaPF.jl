"""
    PowerNetwork

This structure contains constant parameters that define the topology and
physics of the power network.

The object is first created in main memory and then, if GPU computation is
enabled, some of the contents will be moved to the device.

"""
struct PowerNetwork <: AbstractPowerSystem
    vbus::Array{Complex{Float64}}
    Ybus::SparseArrays.SparseMatrixCSC{Complex{Float64},Int64}
    data::Dict{String,Array}

    nbus::Int64
    ngen::Int64

    bustype::Array{Int64}
    bus_to_indexes::Dict{Int, Int}
    ref::Array{Int64}
    pv::Array{Int64}
    pq::Array{Int64}

    sbus::Array{Complex{Float64}}
    sload::Array{Complex{Float64}}

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

get(pf::PowerNetwork, ::NumberOfBuses) = pf.nbus
get(pf::PowerNetwork, ::NumberOfLines) = size(pf.data["branch"], 1)
get(pf::PowerNetwork, ::NumberOfGenerators) = pf.ngen
get(pf::PowerNetwork, ::NumberOfPVBuses) = length(pf.pv)
get(pf::PowerNetwork, ::NumberOfPQBuses) = length(pf.pq)
get(pf::PowerNetwork, ::NumberOfSlackBuses) = length(pf.ref)

function get(pf::PowerNetwork, ::GeneratorIndexes)
    GEN_BUS = IndexSet.idx_gen()[1]
    gens = pf.data["gen"]
    ngens = size(gens)[1]
    indexing = zeros(Int, ngens)
    # Here, we keep the same ordering as specified in Matpower.
    for i in 1:ngens
        indexing[i] = pf.bus_to_indexes[gens[i, GEN_BUS]]
    end
    return indexing
end

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

function print_state(pf::PowerNetwork, x, u, p)
    println("Power Network characteristics:")
    @printf("\tBuses: %d. Slack: %d. PV: %d. PQ: %d\n", pf.nbus, length(pf.ref),
            length(pf.pv), length(pf.pq))
    println("\tGenerators: ", pf.ngen, ".")
    # Print system status
    @printf("\t==============================================\n")
    @printf("\tBUS \t TYPE \t VMAG \t VANG \t P \t Q\n")
    @printf("\t==============================================\n")

    vmag, vang, pinj, qinj = retrieve_physics(pf, x, u, p)

    for i=1:pf.nbus
        type = pf.bustype[i]
        @printf("\t%d \t  %d \t %1.3f\t%3.2f\t%3.3f\t%3.3f\n", i,
                type, vmag[i], vang[i]*(180.0/pi), pinj[i], qinj[i])
    end
end

"""
    get_x(PowerNetwork, vmag, vang, pbus, qbus)

Returns vector x from network variables (VMAG, VANG, P and Q)
and bus type info.

Vector x is the variable vector, consisting on:
    - VMAG, VANG for buses of type PQ (type 1)
    - VANG for buses of type PV (type 2)
These variables are determined by the physics of the network.

Ordering:

x = [VMAG^{PQ}, VANG^{PQ}, VANG^{PV}]
"""
function get_x(
    pf::PowerNetwork,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
) where {T<:Real, VT<:AbstractVector{T}}

    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

    # build vector x
    dimension = 2*npq + npv
    x = zeros(dimension)

    x[1:npq] = vmag[pf.pq]
    x[npq + 1:2*npq] = vang[pf.pq]
    x[2*npq + 1:2*npq + npv] = vang[pf.pv]

    return x
end

"""
    get_u(PowerNetwork, vmag, vang, pbus, qbus)

Returns vector u from network variables (VMAG, VANG, P and Q)
and bus type info.

Vector u is the control vector, consisting on:
    - VMAG, P for buses of type PV (type 1)
    - VM for buses of type SLACK (type 3)
These variables are controlled by the grid operator.

Ordering:

u = [VMAG^{REF}, P^{PV}, VMAG^{PV}]
"""
function get_u(
    pf::PowerNetwork,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
) where {T<:Real, VT<:AbstractVector{T}}

    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

    pload = real.(pf.sload)

    # build vector u
    dimension = 2*npv + nref
    u = zeros(dimension)

    u[1:nref] = vmag[pf.ref]
    # u is equal to active power of generator (Pᵍ)
    # As P = Pᵍ - Pˡ , we get
    u[nref + 1:nref + npv] = pbus[pf.pv] + pload[pf.pv]
    u[nref + npv + 1:nref + 2*npv] = vmag[pf.pv]

    return u
end

"""
    get_p(PowerNetwork, vmag, vang, pbus, qbus)

Returns vector p from network variables (VMAG, VANG, P and Q)
and bus type info.

Vector p is the parameter vector, consisting on:
    - VA for buses of type SLACK (type 3)
    - P, Q for buses of type PQ (type 1)
These parameters are fixed through the problem.

Order:

p = [vang^{ref}, p^{pq}, q^{pq}]
"""
function get_p(
    pf::PowerNetwork,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
) where {T<:Real, VT<:AbstractVector{T}}

    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

    # build vector p
    dimension = nref + 2*npq
    p = zeros(dimension)

    p[1:nref] = vang[pf.ref]
    p[nref + 1:nref + npq] = pbus[pf.pq]
    p[nref + npq + 1:nref + 2*npq] = qbus[pf.pq]

    return p
end

# Some utils function
"""
get_power_injection(fr, v_m, v_a, ybus_re, ybus_im)

Computes the power injection at node "fr".
"""
function get_power_injection(fr, v_m, v_a, ybus_re, ybus_im)
    P = 0.0
    for (j,c) in enumerate(ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1)
        to = ybus_re.rowval[c]
        aij = v_a[fr] - v_a[to]
        P += v_m[fr]*v_m[to]*(ybus_re.nzval[c]*cos(aij) + ybus_im.nzval[c]*sin(aij))
    end
    return P
end

function get_power_injection_partials(fr, v_m, v_a, ybus_re, ybus_im)
    nbus = length(v_m)
    dPdVm = zeros(nbus)
    dPdVa = zeros(nbus)
    for (j,c) in enumerate(ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1)
        to = ybus_re.rowval[c]
        aij = v_a[fr] - v_a[to]

        if to != fr
            dPdVm[to] = v_m[fr]*(ybus_re.nzval[c]*cos(aij) + ybus_im.nzval[c]*sin(aij))
            dPdVa[to] = v_m[fr]*v_m[to]*(ybus_re.nzval[c]*sin(aij) - ybus_im.nzval[c]*cos(aij))
            dPdVm[fr] += v_m[to]*(ybus_re.nzval[c]*cos(aij) + ybus_im.nzval[c]*sin(aij))
            dPdVa[fr] += v_m[to]*v_m[fr]*(-ybus_re.nzval[c]*sin(aij) + ybus_im.nzval[c]*cos(aij))
        else
            dPdVm[fr] += 2*v_m[to]*(ybus_re.nzval[c]*cos(aij) + ybus_im.nzval[c]*sin(aij))
        end
    end
    return dPdVm, dPdVa
end

"""
get_react_injection(fr, v_m, v_a, ybus_re, ybus_im)

Computes the reactive power injection at node "fr".
"""
function get_react_injection(fr, v_m, v_a, ybus_re, ybus_im)
    Q = 0.0
    for (j,c) in enumerate(ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1)
        to = ybus_re.rowval[c]
        aij = v_a[fr] - v_a[to]
        Q += v_m[fr]*v_m[to]*(ybus_re.nzval[c]*sin(aij) - ybus_im.nzval[c]*cos(aij))
    end
    return Q
end

"""
    retrieve_physics(PowerNetwork, x, u, p)

Converts x, u, p vectors to vmag, vang, pinj and qinj.
"""
function retrieve_physics(pf::PowerNetwork, x, u, p; V=Float64)

    nbus = pf.nbus
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

    vmag = zeros(V, nbus)
    vang = zeros(V, nbus)
    pinj = zeros(V, nbus)
    qinj = zeros(V, nbus)

    pload = real.(pf.sload)
    qload = imag.(pf.sload)

    vmag[pf.pq] = x[1:npq]
    vang[pf.pq] = x[npq + 1:2*npq]
    vang[pf.pv] = x[2*npq + 1:2*npq + npv]

    vmag[pf.ref] = u[1:nref]
    # P at buses is equal to Pᵍ (power of generator) minus load Pˡ
    pinj[pf.pv] = u[nref + 1:nref + npv] - pload[pf.pv]
    vmag[pf.pv] = u[nref + npv + 1:nref + 2*npv]

    vang[pf.ref] = p[1:nref]
    pinj[pf.pq] = p[nref + 1:nref + npq]
    qinj[pf.pq] = p[nref + npq + 1:nref + 2*npq]

    # (p, q) for ref and (q) for pv is obtained as a function
    # of the rest of variables
    ybus_re, ybus_im = Spmat{Vector}(pf.Ybus)

    for bus in pf.ref
        pinj[bus] = get_power_injection(bus, vmag, vang, ybus_re, ybus_im)
        qinj[bus] = get_react_injection(bus, vmag, vang, ybus_re, ybus_im)
    end

    for bus in pf.pv
        qinj[bus] = get_react_injection(bus, vmag, vang, ybus_re, ybus_im)
    end

    return vmag, vang, pinj, qinj
end

"""
    get_bound_constraints(pf)

Given PowerNetwork object, returns vectors xmin, xmax, umin, umax
of the OPF box constraints.

"""
function get_bound_constraints(pf::PowerNetwork)

    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()

    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)
    b2i = pf.bus_to_indexes

    gens = pf.data["gen"]
    baseMVA = pf.data["baseMVA"][1]
    bus = pf.data["bus"]
    ngens = size(gens)[1]

    dimension_u = 2*npv + nref
    dimension_x = 2*npq + npv

    u_min = fill(-Inf, dimension_u)
    u_max = fill(Inf, dimension_u)
    x_min = fill(-Inf, dimension_x)
    x_max = fill(Inf, dimension_x)
    p_min = fill(-Inf, nref)
    p_max = fill(Inf, nref)

    for i in 1:length(pf.pq)
        bus_idx = pf.pq[i]
        vm_max = bus[bus_idx, VMAX]
        vm_min = bus[bus_idx, VMIN]
        x_min[i] = vm_min
        x_max[i] = vm_max
    end

    for i in 1:length(pf.pv)
        bus_idx = pf.pv[i]
        vm_max = bus[bus_idx, VMAX]
        vm_min = bus[bus_idx, VMIN]
        u_min[nref + npv + i] = vm_min
        u_max[nref + npv + i] = vm_max
    end

    for i in 1:length(pf.ref)
        bus_idx = pf.ref[i]
        vm_max = bus[bus_idx, VMAX]
        vm_min = bus[bus_idx, VMIN]
        u_min[i] = vm_min
        u_max[i] = vm_max
    end

    for i = 1:ngens
        genbus = b2i[gens[i, GEN_BUS]]
        bustype = bus[genbus, BUS_TYPE]
        if bustype == PV_BUS_TYPE
            idx_pv = findfirst(pf.pv.==genbus)
            u_min[nref + idx_pv] = gens[i, PMIN] / baseMVA
            u_max[nref + idx_pv] = gens[i, PMAX] / baseMVA
        elseif bustype == REF_BUS_TYPE
            idx = findfirst(pf.ref .== genbus)
            p_min[idx] = gens[i, PMIN] / baseMVA
            p_max[idx] = gens[i, PMAX] / baseMVA
        end
    end

    return u_min, u_max, x_min, x_max, p_min, p_max
end

function bounds(pf::PowerNetwork, ::Generator, ::ActivePower)
    gens = pf.data["gen"]
    baseMVA = pf.data["baseMVA"][1]
    # TODO: I think this is wrong
    return  gens[i, PMIN] ./ baseMVA, gens[i, PMAX] ./ baseMVA
end


"""
    get_bound_reactive_power(pf)

Given PowerNetwork object, return bounds on reactive power
of the PV buses.

Matpower specifies bounds on the reactive power of the generator:

    Q_min <= Qᵍ <= Q_max

ExaPF uses internally the reactive power at the buses, which for PV buses
writes out

    Q = Qᵍ - Qˡ

with Qˡ the (constant) reactive load.
This function corrects the bounds to take into account the reactive load:

    Q_min - Qˡ <= Q <= Q_max - Qˡ

"""
function bounds(pf::PowerNetwork, ::Buses, ::ReactivePower)
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()

    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)

    gens = pf.data["gen"]
    baseMVA = pf.data["baseMVA"][1]
    bus = pf.data["bus"]
    ngens = size(gens)[1]
    # Reactive load
    qload = imag.(pf.sload)

    q_min = fill(-Inf, npv)
    q_max = fill(Inf, npv)

    for i = 1:ngens
        genbus = pf.bus_to_indexes[gens[i, GEN_BUS]]
        bustype = bus[genbus, BUS_TYPE]
        if bustype == PV_BUS_TYPE
            idx_pv = findfirst(pf.pv.==genbus)
            q_min[idx_pv] = gens[i, QMIN] / baseMVA - qload[idx_pv]
            q_max[idx_pv] = gens[i, QMAX] / baseMVA - qload[idx_pv]
        end
    end
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
    cost_data = pf.data["cost"]
    ngens = size(gens)[1]
    nbus = size(bus)[1]

    # initialize cost
    # store coefficients in a Float64 array, with 4 columns:
    # - 1st column: bus type
    # - 2nd column: constant coefficient c0
    # - 3rd column: coefficient c1
    # - 4th column: coefficient c2
    coefficients = zeros(ngens, 4)
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

