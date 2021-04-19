function get_inl_sections()
    return ["GOVERNOR PERMANANT DROOP"]
end

function get_inl_section_info()
    governor_droop = [["I", 1, Int], ["ID", 2, String], ["R", 6, Float64]]

    section_info = Dict{String,Array}(
        "GOVERNOR PERMANANT DROOP" => governor_droop)

    return section_info
end

function idx_inl()
    INL_BUS = 1
    INL_ID = 2
    INL_R = 3

    return INL_BUS, INL_ID, INL_R
end

function parse_inl(filename::AbstractString; delim=",")
    f = open(filename, "r")
    lines = readlines(f)
    close(f)

    sections = get_inl_sections()
    section_info = get_inl_section_info()
    pos = 1

    inldata = Dict{String,Array}()

    # There is only one section in case.inl file.
    s = sections[1]
    pos, data = parse_data(lines, pos, section_info[s])
    inldata[s] = data

    return inldata
end

