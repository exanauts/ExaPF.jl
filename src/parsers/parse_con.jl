function idx_con()
    CON_LABEL = 1
    CON_TYPE = 2
    CON_FR = 3
    CON_TO = 4
    CON_ID = 5

    return CON_LABEL, CON_TYPE, CON_FR, CON_TO, CON_ID
end

function parse_con(filename::AbstractString; delim=" \t")
    f = open(filename, "r")
    lines = readlines(f)
    close(f)

    pos = 1
    ndata = 0
    while pos <= length(lines)
        line = strip(lines[pos])

        if line[1] == 'R' || line[1] == 'O'
            ndata += 1
        end

        pos += 1
    end

    data = Array{Any}(undef, ndata, 5)

    label = ""
    ndata = 0
    pos = 1
    while pos <= length(lines)
        line = strip(lines[pos])
        fields = split(line)

        if line[1] == 'C'      # CONTINGENCY
            label = fields[2]
        elseif line[1] == 'R'  # REMOVE
            ndata += 1
            data[ndata,1] = label
            data[ndata,2] = 2                     # ctg type: generator
            data[ndata,3] = parse(Int, fields[6]) # I
            data[ndata,4] = 0                     # No J for generator
            data[ndata,5] = fields[3]             # ID
        elseif line[1] == 'O'  # OPEN
            ndata += 1
            data[ndata,1] = label
            data[ndata,2] = 1                     # ctg type: branch
            data[ndata,3] = parse(Int, fields[5]) # I
            data[ndata,4] = parse(Int, fields[8]) # J
            data[ndata,5] = fields[10]            # CKT
        end

        pos += 1
    end

    condata = Dict{String,Array}()
    condata["CONTINGENCY"] = data

    return condata
end
