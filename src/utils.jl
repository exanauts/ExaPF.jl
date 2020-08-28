struct ConvergenceStatus
    has_converged::Bool
    n_iterations::Int
    norm_residuals::Float64
    n_linear_solves::Int
end

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

# small utils function
function polar!(Vm, Va, V, ::CPU)
    Vm .= abs.(V)
    Va .= angle.(V)
end
function polar!(Vm, Va, V, ::CUDADevice)
    Vm .= CUDA.abs.(V)
    Va .= CUDA.angle.(V)
end
