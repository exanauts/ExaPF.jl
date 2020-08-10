module ParsePSSE

function parse_skipsection(lines::Array{String}, pos::Int)
    while pos <= length(lines)
        line = strip(lines[pos])

        if line[1] in [ '0', 'Q' ]
            break
        end

        pos += 1
    end

    return pos
end

function parse_data(lines::Array{String}, pos::Int, info::Array;
                  delim=",")
    start = last = pos
    last = parse_skipsection(lines, start)
    ndata = last - start
    data = Array{Any}(undef, ndata, length(info))

    for i=1:ndata
        # Get rid of the single quotes.
        line = replace(lines[start+i-1], "'" => "")
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

include("parse_raw.jl")
include("parse_rop.jl")
include("parse_inl.jl")
include("parse_con.jl")

end
