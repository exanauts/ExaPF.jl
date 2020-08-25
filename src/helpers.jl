mutable struct Spmat{T}
    colptr
    rowval
    nzval

    # create 2 Spmats from complex matrix
    function Spmat{T}(mat::SparseMatrixCSC{Complex{Float64}, Int}) where T
        matreal = new(T{Int64}(mat.colptr), T{Int64}(mat.rowval), T{Float64}(real.(mat.nzval)))
        matimag = new(T{Int64}(mat.colptr), T{Int64}(mat.rowval), T{Float64}(imag.(mat.nzval)))
        return matreal, matimag
    end
    # copy constructor
    function Spmat{T}(mat) where T
        return new(T{Int64}(mat.colptr), T{Int64}(mat.rowval), T{Float64}(mat.nzval))
    end
end
