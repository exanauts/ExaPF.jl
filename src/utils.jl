struct ConvergenceStatus
    has_converged::Bool
    n_iterations::Int
    norm_residuals::Float64
    n_linear_solves::Int
end

mutable struct Spmat{VTI<:AbstractVector, VTF<:AbstractVector}
    colptr::VTI
    rowval::VTI
    nzval::VTF

    # create 2 Spmats from complex matrix
    function Spmat{VTI, VTF}(mat::SparseMatrixCSC{Complex{Float64}, Int}) where {VTI, VTF}
        matreal = new(VTI(mat.colptr), VTI(mat.rowval), VTF(real.(mat.nzval)))
        matimag = new(VTI(mat.colptr), VTI(mat.rowval), VTF(imag.(mat.nzval)))
        return matreal, matimag
    end
    # copy constructor
    function Spmat{VTI, VTF}(mat) where {VTI, VTF}
        return new(VTI(mat.colptr), VTI(mat.rowval), VTF(mat.nzval))
    end
end

# norm
norm2(x::AbstractVector) = norm(x, 2)
norm2(x::CuVector) = CUBLAS.nrm2(x)

function project_constraints!(u::AbstractArray, grad::AbstractArray, u_min::AbstractArray,
                              u_max::AbstractArray)
    dim = length(u)
    for i in 1:dim
        if u[i] > u_max[i]
            u[i] = u_max[i]
            grad[i] = 0.0
        elseif u[i] < u_min[i]
            u[i] = u_min[i]
            grad[i] = 0.0
        end
    end
end

# Utils function to solve transposed linear system  A' x = y
# Source code taken from:
# https://github.com/JuliaGPU/CUDA.jl/blob/master/lib/cusolver/wrappers.jl#L78L111
function csclsvqr!(A::CuSparseMatrixCSC{Float64},
                    b::CuVector{Float64},
                    x::CuVector{Float64},
                    tol::Float64,
                    reorder::Cint,
                    inda::Char)
    n = size(A,1)
    desca = CUSPARSE.CuMatrixDescriptor(
        CUSPARSE.CUSPARSE_MATRIX_TYPE_GENERAL,
        CUSPARSE.CUSPARSE_FILL_MODE_LOWER,
        CUSPARSE.CUSPARSE_DIAG_TYPE_NON_UNIT, inda)
    singularity = Ref{Cint}(1)
    CUSOLVER.cusolverSpDcsrlsvqr(CUSOLVER.sparse_handle(), n, A.nnz, desca, A.nzVal, A.colPtr, A.rowVal, b, tol, reorder, x, singularity)

    if singularity[] != -1
        throw(SingularException(singularity[]))
    end

    x
end
