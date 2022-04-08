function convert2matpower(data_mat::Dict{String, Any})
    baseMVA       = convert(Float64, data_mat["baseMVA"])
    buses_dict    = data_mat["bus"]
    gens_dict     = data_mat["gen"]
    branches_dict = data_mat["branch"]
    shunt_dict    = data_mat["shunt"]
    load_dict     = data_mat["load"]

    # buses
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()

    nbus = length(buses_dict)
    bus_array = zeros(nbus, 13)

    # Keep the correspondence between Matpower's ID and ExaPF contiguous indexing.
    for (_, bus) in buses_dict
        i = bus["bus_i"]::Int
        bus_array[i, BUS_I] = i
        bus_array[i, BUS_TYPE] = bus["bus_type"]
        bus_array[i, BUS_AREA] = bus["area"]
        bus_array[i, VM] = bus["vm"]
        bus_array[i, VA] = bus["va"]
        bus_array[i, BASE_KV] = bus["base_kv"]
        bus_array[i, ZONE] = bus["zone"]
        bus_array[i, VMAX] = bus["vmax"]
        bus_array[i, VMIN] = bus["vmin"]
    end
    # Loads
    for (_, load) in load_dict
        i = load["load_bus"]::Int
        bus_array[i, PD] = load["pd"]
        bus_array[i, QD] = load["qd"]
    end
    # Shunts
    for (_, shunt) in shunt_dict
        i = shunt["shunt_bus"]::Int
        bus_array[i, GS] = shunt["gs"]
        bus_array[i, BS] = shunt["bs"]
    end

    # generators
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN,
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAG, MU_PMIN, MU_QMAX,
    MU_QMIN = IndexSet.idx_gen()

    ngen = length(gens_dict)
    gen_array = zeros(ngen, 16)

    for (_, gen) in gens_dict
        i = gen["index"]::Int
        gen_array[i, GEN_BUS] = gen["gen_bus"]
        gen_array[i, PG] = gen["pg"]
        gen_array[i, QG] = gen["qg"]
        gen_array[i, QMAX] = gen["qmax"]
        gen_array[i, QMIN] = gen["qmin"]
        gen_array[i, VG] = gen["vg"]
        gen_array[i, MBASE] = gen["mbase"]
        gen_array[i, GEN_STATUS] = gen["gen_status"]
        gen_array[i, PMAX] = gen["pmax"]
        gen_array[i, PMIN] = gen["pmin"]
        # pc1, pc2, qc1min, qc2min could be not specified.
        # Set default value to 0.
        gen_array[i, PC1] = get(gen, "pc1", 0.0)
        gen_array[i, PC2] = get(gen, "pc2", 0.0)
        gen_array[i, QC1MIN] = get(gen, "qc1min", 0.0)
        gen_array[i, QC2MIN] = get(gen, "qc2min", 0.0)
    end

    F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS,
    ANGMIN, ANGMAX, PF, QF, PT, QT, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX = IndexSet.idx_branch()

    nbranch = length(branches_dict)
    branch_array = zeros(nbranch, 13)

    for (_, branch) in branches_dict
        i = branch["index"]::Int
        branch_array[i, F_BUS] = branch["f_bus"]
        branch_array[i, T_BUS] = branch["t_bus"]
        branch_array[i, BR_R] = branch["br_r"]
        branch_array[i, BR_X] = branch["br_x"]
        branch_array[i, BR_B] = branch["b_fr"] + branch["b_to"]
        branch_array[i, RATE_A] = get(branch, "rate_a", 0.0)
        branch_array[i, RATE_B] = get(branch, "rate_b", 0.0)
        branch_array[i, RATE_C] = get(branch, "rate_c", 0.0)
        branch_array[i, TAP] = branch["tap"]
        branch_array[i, SHIFT] = branch["shift"]
        branch_array[i, BR_STATUS] = branch["br_status"]
        branch_array[i, ANGMIN] = branch["angmin"]
        branch_array[i, ANGMAX] = branch["angmax"]
        #branch_array[i, PF] = branch["pf"]
        #branch_array[i, QF] = branch["qf"]
        #branch_array[i, PT] = branch["pt"]
        #branch_array[i, QT] = branch["qt"]
    end

    MODEL, STARTUP, SHUTDOWN, NCOST, COST = IndexSet.idx_cost()
    ncost = length(gens_dict)
    cost_array = zeros(ncost, 7)
    for (_, gen) in gens_dict
        i = gen["index"]::Int
        cost_array[i, MODEL] = gen["model"]
        cost_array[i, STARTUP] = gen["startup"]
        cost_array[i, SHUTDOWN] = gen["shutdown"]
        ncosts = gen["ncost"]::Int
        cost_array[i, NCOST] = ncosts
        for j in 1:ncosts
            cost_array[i, NCOST+j] = gen["cost"][j]
        end
    end

    return bus_array, branch_array, gen_array, cost_array, baseMVA
end
