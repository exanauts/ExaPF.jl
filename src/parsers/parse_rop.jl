function get_rop_sections()
    return ["MODIFICATION CODE", "BUS VOLTAGE", "ADJUSTABLE BUS SHUNTS",
            "BUS LOADS", "ADJUSTABLE BUS LOAD", "GENERATOR DISPATCH",
            "ACTIVE POWER DISPATCH", "GENERATOR RESERVE",
            "GENERATION REACTIVE CAP", "ADJUSTABLE BRANCH",
            "PIECEWISE LINEAR COST", "PIECEWISE QUADRATIC COST",
            "POLYNOMIAL EXPONENTIAL COST", "PERIOD RESERVES",
            "BRANCH FLOWS", "INTERFACE FLOWS", "LINEAR CONSTRAINT DEP"]
end

function get_rop_section_info()
    generator_dispatch = [["BUS", 1, Int], ["GENID", 2, String],
                          ["DSPTBL", 4, Int]]
    active_power_dispatch = [["TBL", 1, Int], ["CTBL", 7, Int]]
    piecewise_linear_cost = [["LTBL", 1, Int], ["NPARIS", 3, Int]]

    section_info = Dict{String,Array}(
        "MODIFICATION CODE" => [],
        "BUS VOLTAGE" => [],
        "ADJUSTABLE BUS SHUNTS" => [],
        "BUS LOADS" => [],
        "ADJUSTABLE BUS LOAD" => [],
        "GENERATOR DISPATCH" => generator_dispatch,
        "ACTIVE POWER DISPATCH" => active_power_dispatch,
        "GENERATOR RESERVE" => [],
        "GENERATION REACTIVE CAP" => [],
        "ADJUSTABLE BRANCH" => [],
        "PIECEWISE LINEAR COST" => piecewise_linear_cost,
        "PIECEWISE QUADRATIC COST" => [],
        "POLYNOMIAL EXPONENTIAL COST" => [],
        "PERIOD RESERVES" => [],
        "BRANCH FLOWS" => [],
        "INTERFACE FLOWS" => [],
        "LINEAR CONSTRAINT DEP" => [])

    return section_info
end

function idx_rop()
    ROP_BUS = 1
    ROP_ID = 2
    ROP_NPAIRS = 3
    ROP_PAIRSTART = 4

    return ROP_BUS, ROP_ID, ROP_NPAIRS, ROP_PAIRSTART
end

function rop_join_tables(ropdata::Dict{String,Array})
    gendspt = ropdata["GENERATOR DISPATCH"]
    acpdspt = ropdata["ACTIVE POWER DISPATCH"]
    pwlcost = ropdata["PIECEWISE LINEAR COST"]

    ndata = size(gendspt, 1)
    data = Array{Any}(undef, ndata, 4)

    for i=1:ndata
        dsptbl = gendspt[i, 3]

        if dsptbl <= ndata && dsptbl == acpdspt[dsptbl, 1]
            acp_row = dsptbl
        else
            acp_row = findfirst(dsptbl .== acpdspt[:, 1])
        end

        if acpdspt[acp_row, 2] == pwlcost[acp_row, 1]
            pwl_row = acp_row
        else
            pwl_row = findfirst(acpdspt[acp_row, 2] .== pwlcost[:, 1])
        end

        data[i, 1] = gendspt[i, 1]
        data[i, 2] = gendspt[i, 2]
        data[i, 3] = pwlcost[pwl_row, 2]
        data[i, 4] = pwlcost[pwl_row, 3]
    end

    return data
end

function rop_pwl(lines::Array{String}, pos::Int, info::Array;
                 delim=",")
    NPAIRS_LOC = 2
    start = last = pos
    ndata = npairs = tot_npairs = 0

    line = strip(lines[last])
    while line[1] ∉ [ '0', 'Q' ]
        fields = split(line, delim)
        npairs = parse(info[2][3], fields[info[2][2]])
        tot_npairs += npairs
        ndata += 1
        last += 1 + npairs
        line = strip(lines[last])
    end

    data = Array{Any}(undef, ndata, length(info)+1)
    pairs = Array{Float64}(undef, tot_npairs, 2)

    offset = start
    npairs = tot_npairs = 0
    for i=1:ndata
        fields = split(lines[offset], delim)

        for j=1:length(info)
            data[i,j] = parse(info[j][3], fields[info[j][2]])
        end

        offset += 1
        npairs = data[i,NPAIRS_LOC]
        data[i,end] = tot_npairs
        for j=1:npairs
            tot_npairs += 1
            fields = split(lines[offset], delim)
            pairs[tot_npairs,1] = parse(Float64, fields[1])
            pairs[tot_npairs,2] = parse(Float64, fields[2])
            offset += 1
        end
    end

    pos = last + 1
    return pos, data, pairs
end

function parse_rop(filename::AbstractString; delim=",")
    f = open(filename, "r")
    lines = readlines(f)
    close(f)

    sections = get_rop_sections()
    section_info = get_rop_section_info()
    pos = 1

    ropdata = Dict{String,Array}()
    for s in sections
        info = section_info[s]

        if isempty(info)
            pos = parse_skipsection(lines, pos) + 1
            continue
        end

        if s == "PIECEWISE LINEAR COST"
            pos, data, pairs = rop_pwl(lines, pos, info; delim=delim)
            ropdata["PAIRS"] = pairs
        else
            pos, data = parse_data(lines, pos, info; delim=delim)
        end

        ropdata[s] = data
    end

    # Perform join operation
    if haskey(ropdata, "PIECEWISE LINEAR COST")
        data = rop_join_tables(ropdata)
        ropdata["JOINED GENERATOR DISPATCH"] = data
    end

    return ropdata
end
