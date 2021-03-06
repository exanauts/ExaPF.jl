export NewtonRaphson

abstract type AbstractNonLinearSolver end

"""
    NewtonRaphson <: AbstractNonLinearSolver

Newton-Raphson algorithm. Used to solve the non-linear equation
``g(x, u) = 0``, at a fixed control ``u``.

### Attributes
- `maxiter::Int` (default 20): maximum number of iterations
- `tol::Float64` (default `1e-8`): tolerance of the algorithm
- `verbose::Int` (default `NONE`): verbosity level

"""
struct NewtonRaphson <: AbstractNonLinearSolver
    maxiter::Int
    tol::Float64
    verbose::Int
end
NewtonRaphson(; maxiter=20, tol=1e-8, verbose=VERBOSE_LEVEL_NONE) = NewtonRaphson(maxiter, tol, verbose)

"""
    ConvergenceStatus

Convergence status returned by a non-linear algorithm.

### Attributes
- `has_converged::Bool`: states whether the algorithm has converged.
- `n_iterations::Int`: total number of iterations of the non-linear algorithm.
- `norm_residuals::Float64`: final residual.
- `n_linear_solves::Int`: number of linear systems ``Ax = b`` resolved during the run.

"""
struct ConvergenceStatus
    has_converged::Bool
    n_iterations::Int
    norm_residuals::Float64
    n_linear_solves::Int
end

# Sparse utilities
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

function _copy_csc!(J_dest, J_src, shift)
    @inbounds for i in 1:size(J_src, 2)
        for j in J_src.colptr[i]:J_src.colptr[i+1]-1
            row = J_src.rowval[j]
            @inbounds J_dest[row+shift, i] = J_src.nzval[j]
        end
    end
end

function _transfer_sparse!(J_dest::SparseMatrixCSC, J_src::SparseMatrixCSC, shift)
    _copy_csc!(J_dest, J_src, shift)
end

function _transfer_sparse!(J_dest::CUSPARSE.CuSparseMatrixCSR, J_src::CUSPARSE.CuSparseMatrixCSR, shift)
    nnz_start = J_dest.rowPtr[shift+1]
    nnz_ = nnz(J_src)
    Jnnz = @view J_dest.nzVal[nnz_start:nnz_start+nnz_-1]
    Jnnz .= J_src.nzVal
end


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

