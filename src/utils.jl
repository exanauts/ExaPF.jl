export NewtonRaphson

abstract type AbstractNonLinearSolver end

struct NewtonRaphson <: AbstractNonLinearSolver
    maxiter::Int
    tol::Float64
    verbose::Int
end
NewtonRaphson(; maxiter=20, tol=1e-8, verbose=VERBOSE_LEVEL_NONE) = NewtonRaphson(maxiter, tol, verbose)

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
xnorm(x::AbstractVector) = norm(x, 2)
xnorm(x::CUDA.CuVector) = CUBLAS.nrm2(x)

# Array initialization
xzeros(S, n) = fill!(S(undef, n), zero(eltype(S)))
xones(S, n) = fill!(S(undef, n), one(eltype(S)))

# projection operator
function project!(w::VT, u::VT, u♭::VT, u♯::VT) where VT<:AbstractArray
    w .= max.(min.(u, u♯), u♭)
end

# Utils function to solve transposed linear system  A' x = y
# Source code taken from:
# https://github.com/JuliaGPU/CUDA.jl/blob/master/lib/cusolver/wrappers.jl#L78L111
function csclsvqr!(A::CUSPARSE.CuSparseMatrixCSC{Float64},
                    b::CUDA.CuVector{Float64},
                    x::CUDA.CuVector{Float64},
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

