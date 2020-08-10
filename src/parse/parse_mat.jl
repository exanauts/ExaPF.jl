module ParseMAT

function parse_mat(filename::AbstractString; delim=",")
    f = open(filename, "r")
    lines = readlines(f)
    close(f)

    println(lines)

    for line in lines
        println(lines)
    end

end

end
