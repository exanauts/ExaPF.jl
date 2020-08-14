function get_raw_sections()
    return ["CASE IDENTIFICATION", "BUS", "LOAD", "FIXED SHUNT",
            "GENERATOR", "BRANCH", "TRANSFORMER", "AREA INTERCHANGE",
            "TWO-TERMINAL DC", "VOLTAGE SOURCE CONVERTER",
            "IMPEDANCE CORRECTION", "MULTI-TERMINAL DC",
            "MULTI-SECTION LINE", "ZONE", "INTER-AREA TRANSFER", "OWNER",
            "FACTS CONTROL DEVICE", "SWITCHED SHUNT", "GNE DEVICE",
            "INDUCTION MACHINE"]
end

function get_raw_section_info()
    case_identification = [["SBASE", 2, Float64]]
    bus = [["I", 1, Int], ["AREA", 5, Int], ["VM", 8, Float64],
           ["VA", 9, Float64], ["NVHI", 10, Float64], ["NVLO", 11, Float64],
           ["EVHI", 12, Float64], ["EVLO", 13, Float64], ["TYPE", 4, Int]]
    load = [["I", 1, Int], ["ID", 2, String], ["STATUS", 3, Int],
            ["PL", 6, Float64], ["QL", 7, Float64]]
    fixed_shunt = [["I", 1, Int], ["ID", 2, String], ["STATUS", 3, Int],
                   ["GL", 4, Float64], ["BL", 5, Float64]]
    generator = [["I", 1, Int], ["ID", 2, String], ["PG", 3, Float64],
                 ["QG", 4, Float64], ["QT", 5, Float64], ["QB", 6, Float64],
                 ["STAT", 15, Int], ["PT", 17, Float64], ["PB", 18, Float64]]
    branch = [["I", 1, Int], ["J", 2, Int], ["CKT", 3, String],
              ["R", 4, Float64], ["X", 5, Float64], ["B", 6, Float64],
              ["RATEA", 7, Float64], ["RATEC", 9, Float64], ["ST", 14, Int]]
    transformer = [["I", 1, Int], ["J", 2, Int], ["CKT", 4, String],
                   ["MAG1", 8, Float64], ["MAG2", 9, Float64],
                   ["STAT", 12, Int], ["R12", 22, Float64],
                   ["X12", 23, Float64], ["WINDV1", 25, Float64],
                   ["ANG1", 27, Float64], ["RATA1", 28, Float64],
                   ["RATC1", 30, Float64], ["WINDV2", 42, Float64]]
    switched_shunt = [["I", 1, Int], ["STAT", 4, Int], ["BINIT", 10, Float64],
                      ["N1", 11, Float64], ["B1", 12, Float64],
                      ["N2", 13, Float64], ["B2", 14, Float64],
                      ["N3", 15, Float64], ["B3", 16, Float64],
                      ["N4", 17, Float64], ["B4", 18, Float64],
                      ["N5", 19, Float64], ["B5", 20, Float64],
                      ["N6", 21, Float64], ["B6", 22, Float64],
                      ["N7", 23, Float64], ["B7", 24, Float64],
                      ["N8", 25, Float64], ["B8", 26, Float64]]

    section_info = Dict{String,Array}(
        "CASE IDENTIFICATION" => case_identification,
        "BUS" => bus,
        "LOAD" => load,
        "FIXED SHUNT" => fixed_shunt,
        "GENERATOR" => generator,
        "BRANCH" => branch,
        "TRANSFORMER" => transformer,
        "AREA INTERCHANGE" => [],
        "TWO-TERMINAL DC" => [],
        "VOLTAGE SOURCE CONVERTER" => [],
        "IMPEDANCE CORRECTION" => [],
        "MULTI-TERMINAL DC" => [],
        "MULTI-SECTION LINE" => [],
        "ZONE" => [],
        "INTER-AREA TRANSFER" => [],
        "OWNER" => [],
        "FACTS CONTROL DEVICE" => [],
        "SWITCHED SHUNT" => switched_shunt,
        "GNE DEVICE" => [],
        "INDUCTION MACHINE" => [])

    return section_info
end

function idx_bus()
    BUS_B = 1
    BUS_AREA = 2
    BUS_VM = 3
    BUS_VA = 4
    BUS_NVHI = 5
    BUS_NVLO = 6
    BUS_EVHI = 7
    BUS_EVLO = 8
    BUS_TYPE = 9

    return BUS_B, BUS_AREA, BUS_VM, BUS_VA, BUS_NVHI, BUS_NVLO, BUS_EVHI, BUS_EVLO, BUS_TYPE
end

function idx_gen()
    GEN_BUS = 1
    GEN_ID = 2
    GEN_PG = 3
    GEN_QG = 4
    GEN_QT = 5
    GEN_QB = 6
    GEN_STAT = 7
    GEN_PT = 8
    GEN_PB = 9

    return GEN_BUS, GEN_ID, GEN_PG, GEN_QG, GEN_QT, GEN_QB, GEN_STAT, GEN_PT, GEN_PB
end

function idx_load()
    LOAD_BUS = 1
    LOAD_ID = 2
    LOAD_STAT = 3
    LOAD_PL = 4
    LOAD_QL = 5
    return LOAD_BUS, LOAD_ID, LOAD_STAT, LOAD_PL, LOAD_QL
end

function idx_branch()
    BR_FR = 1
    BR_TO = 2
    BR_ID = 3
    BR_R = 4
    BR_X = 5
    BR_B = 6
    BR_RATEA = 7
    BR_RATEC = 8
    BR_STAT = 9

    return BR_FR, BR_TO, BR_ID, BR_R, BR_X, BR_B, BR_RATEA, BR_RATEC, BR_STAT
end

function idx_fshunt()
    FSH_BUS = 1
    FSH_ID = 2
    FSH_STAT = 3
    FSH_G = 4
    FSH_B = 5

    return FSH_BUS, FSH_ID, FSH_STAT, FSH_G, FSH_B
end

function idx_transformer()
    TR_FR = 1
    TR_TO = 2
    TR_ID = 3
    TR_MAG1 = 4
    TR_MAG2 = 5
    TR_STAT = 6
    TR_R = 7
    TR_X = 8
    TR_WINDV1 = 9
    TR_ANG = 10
    TR_RATEA = 11
    TR_RATEC = 12
    TR_WINDV2 = 13

    return TR_FR, TR_TO, TR_ID, TR_MAG1, TR_MAG2, TR_STAT, TR_R, TR_X,
           TR_WINDV1, TR_ANG, TR_RATEA, TR_RATEC, TR_WINDV2
end

function idx_sshunt()
    SSH_BUS = 1
    SSH_STAT = 2
    SSH_BINIT = 3
    SSH_N1 = 4
    SSH_B1 = 5

    return SSH_BUS, SSH_STAT, SSH_BINIT, SSH_N1, SSH_B1
end

function raw_xfmr(lines::Array{String}, pos::Int, info::Array;
                  delim=",")
    start = last = pos
    while strip(lines[last])[1] ∉ [ '0', 'Q' ]
        last += 4
    end

    ndata = Int((last - start)/4)
    data = Array{Any}(undef, ndata, length(info))

    for i=1:ndata
        # Each transformer record consists of 4 lines.
        # We read them all and make a single line.
        offset = 4*(i-1)
        line = replace(lines[start + offset], "'" => "")
        for j=1:3
            line *= ","*replace(lines[start + offset + j], "'" => "")
        end

        fields = split(line, delim)
        for j=1:length(info)
            if info[j][3] == String
                data[i,j] = strip(fields[info[j][2]])
            else
                data[i,j] = parse(info[j][3], fields[info[j][2]])
            end
        end
    end

    pos = last + 1
    return pos, data
end

function parse_raw(filename::AbstractString; delim=",")
    f = open(filename, "r")
    lines = readlines(f)
    close(f)

    sections = get_raw_sections()
    section_info = get_raw_section_info()
    pos = 1

    rawdata = Dict{String,Array}()
    for s in sections
        info = section_info[s]

        if isempty(info)
            pos = parse_skipsection(lines, pos) + 1
            continue
        end

        if s == "CASE IDENTIFICATION"
            line = lines[pos]
            data = [parse(Float64, split(line, delim)[info[1][2]])]
            pos += 3
        elseif s == "TRANSFORMER"
            pos, data = raw_xfmr(lines, pos, info; delim=delim)
        else
            pos, data = parse_data(lines, pos, info; delim=delim)
        end

        rawdata[s] = data
    end

    return rawdata
end

function raw_to_exapf(rawdata::Dict{String, Array})

    BUS_B, BUS_AREA, BUS_VM, BUS_VA, BUS_NVHI, BUS_NVLO, BUS_EVHI,
    BUS_EVLO, BUS_TYPE = idx_bus()
    GEN_BUS, GEN_ID, GEN_PG, GEN_QG, GEN_QT, GEN_QB, GEN_STAT,
    GEN_PT, GEN_PB = idx_gen()
    LOAD_BUS, LOAD_ID, LOAD_STATUS, LOAD_PL, LOAD_QL = idx_load()
    BR_FR, BR_TO, BR_CKT, BR_R, BR_X, BR_B, BR_RATEA, BR_RATEC,
        BR_STAT = idx_branch()
    TR_FR, TR_TO, TR_CKT, TR_MAG1, TR_MAG2, TR_STAT, TR_R, TR_X, TR_WINDV1,
        TR_ANG, TR_RATEA, TR_RATEC, TR_WINDV2 = idx_transformer()
    FSH_BUS, FSH_ID, FSH_STAT, FSH_G, FSH_B = idx_fshunt()

    data = Dict{String,Array}()
    bus = rawdata["BUS"]
    gen = rawdata["GENERATOR"]
    load = rawdata["LOAD"]
    branch = rawdata["BRANCH"]
    trans = rawdata["TRANSFORMER"]
    fsh = rawdata["FIXED SHUNT"]
    SBASE = rawdata["CASE IDENTIFICATION"][1]

    nbus = size(bus, 1)
    ngen = size(gen, 1)
    nload = size(load, 1)
    nbranch = size(branch, 1)
    ntrans = size(trans, 1)
    nfsh = size(fsh, 1)

    # create new data structure
    data = Dict{String,Array}()

    # perunit base
    data["baseMVA"] = [SBASE]

    # buses
    bus_array = zeros(nbus, 17)
    for i in 1:nbus
        bus_array[i, 1] = bus[i, BUS_B]
        bus_array[i, 2] = bus[i, BUS_TYPE]
        bus_array[i, 3] = 0.0
        bus_array[i, 4] = 0.0
        bus_array[i, 5] = 0.0
        bus_array[i, 6] = 0.0
        bus_array[i, 7] = bus[i, BUS_AREA]
        bus_array[i, 8] = bus[i, BUS_VM]
        bus_array[i, 9] = bus[i, BUS_VA]
    end

    # fixed shunt
    for i in 1:nfsh
        if fsh[i, FSH_STAT] == 0
            break
        end
        bus_idx = fsh[i, FSH_BUS]
        bus_array[bus_idx, 5] += fsh[i, FSH_G]
        bus_array[bus_idx, 6] += fsh[i, FSH_B]
    end

    # add load
    for i in 1:nload
        if load[i, LOAD_STATUS] == 0
            break
        end
        bus_idx = load[i, LOAD_BUS]
        Pd = load[i, LOAD_PL]
        Qd = load[i, LOAD_QL]
        bus_array[bus_idx, 3] += Pd
        bus_array[bus_idx, 4] += Qd
    end
    data["bus"] = bus_array

    # gens
    gen_array = zeros(ngen, 25)
    for i in 1:ngen
        gen_array[i, 1] = gen[i, GEN_BUS]
        gen_array[i, 2] = gen[i, GEN_PG]
        gen_array[i, 3] = gen[i, GEN_QG]
        gen_array[i, 8] = gen[i, GEN_STAT]
    end
    data["gen"] = gen_array

    # branches
    branch_array = zeros(nbranch + ntrans, 21)
    for i in 1:nbranch
        branch_array[i, 1] = branch[i, BR_FR]
        branch_array[i, 2] = branch[i, BR_TO]
        branch_array[i, 3] = branch[i, BR_R]
        branch_array[i, 4] = branch[i, BR_X]
        branch_array[i, 5] = branch[i, BR_B]
        branch_array[i, 9] = 0.0
        branch_array[i, 10] = 0.0
        branch_array[i, 11] = branch[i, BR_STAT]
    end

    for i in 1:ntrans
        branch_array[nbranch + i, 1] = trans[i, TR_FR]
        branch_array[nbranch + i, 2] = trans[i, TR_TO]
        branch_array[nbranch + i, 3] = trans[i, TR_R]
        branch_array[nbranch + i, 4] = trans[i, TR_X]
        branch_array[nbranch + i, 5] = 0.0
        tap = trans[i, TR_WINDV1] / trans[i, TR_WINDV2]
        branch_array[nbranch + i, 9] = tap
        branch_array[nbranch + i, 10] = trans[i, TR_ANG]
        branch_array[nbranch + i, 11] = trans[i, TR_STAT]
    end
    data["branch"] = branch_array

    # TODO:
    bus_id_to_indexes = Dict{Int, Int}([(i, i) for i in 1:nbus])
    return data, bus_id_to_indexes
end

