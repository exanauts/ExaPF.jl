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
end

mutable struct BatchCuSparseMatrixCSR{Tv} <: CUSPARSE.AbstractCuSparseMatrix{Tv}
    rowPtr::CUDA.CuArray{Cint, 1, CUDA.Mem.DeviceBuffer}
    colVal::CUDA.CuArray{Cint, 1, CUDA.Mem.DeviceBuffer}
    nzVal::CUDA.CuArray{Tv, 2, CUDA.Mem.DeviceBuffer}
    dims::NTuple{2,Int}
    nnz::Cint
    nbatch::Int

    function BatchCuSparseMatrixCSR{Tv}(rowPtr::CUDA.CuArray{<:Integer, 1, CUDA.Mem.DeviceBuffer}, colVal::CUDA.CuArray{<:Integer, 1, CUDA.Mem.DeviceBuffer},
                                   nzVal::CUDA.CuMatrix, dims::NTuple{2,<:Integer}, nnzJ::Int, nbatch::Int) where Tv
        new(rowPtr, colVal, nzVal, dims, nnzJ, nbatch)
    end
end

Base.size(J::BatchCuSparseMatrixCSR) = J.dims
function BatchCuSparseMatrixCSR(J::SparseMatrixCSC{Tv, Int}, nbatch) where Tv
    dims = size(J)
    nnzJ = nnz(J)
    d_J = CUSPARSE.CuSparseMatrixCSR(J)
    nzVal = CUDA.zeros(Tv, nnzJ, nbatch)
    for i in 1:nbatch
        copyto!(nzVal, nnzJ * (i-1) + 1, J.nzval, 1, nnzJ)
    end
    return BatchCuSparseMatrixCSR{Tv}(d_J.rowPtr, d_J.colVal, nzVal, dims, nnzJ, nbatch)
end

function CUDA.unsafe_free!(xs::BatchCuSparseMatrixCSR)
    unsafe_free!(xs.rowPtr)
    unsafe_free!(xs.colVal)
    unsafe_free!(xs.nzVal)
    return
end

function _copy_csc!(J_dest, J_src, shift)
    @inbounds for i in 1:size(J_src, 2)
        for j in J_src.colptr[i]:J_src.colptr[i+1]-1
            row = J_src.rowval[j]
            @inbounds J_dest[row+shift, i] = J_src.nzval[j]
        end
    end
end

function _transfer_sparse!(J_dest::SparseMatrixCSC, J_src::SparseMatrixCSC, shift, device)
    _copy_csc!(J_dest, J_src, shift)
end

@kernel function _copy_sparse_matric_csr!(Jdest, Jsrc, rowptr, shift, nnz_)
    i = @index(Global, Linear)
    nnz_start = rowptr[shift+1] - 1
    # Jnnz = @view J_dest.nzVal[nnz_start:nnz_start+nnz_-1]
    Jdest[i+nnz_start] = Jsrc[i]
end

function _transfer_sparse!(J_dest::CUSPARSE.CuSparseMatrixCSR, J_src::CUSPARSE.CuSparseMatrixCSR, shift, device)
    nnz_ = nnz(J_src)
    _copy_sparse_matric_csr!(device)(
        J_dest.nzVal, J_src.nzVal, J_dest.rowPtr, shift, nnz_,
        ndrange=nnz_,
    )
end

# Utils function to solve transposed linear system  A' x = y
# Source code taken from:
# https://github.com/JuliaGPU/CUDA.jl/blob/master/lib/cusolver/wrappers.jl#L78L111
function csclsvqr!(A::CUSPARSE.CuSparseMatrixCSC{Float64},
                    b::CUDA.CuArray{Float64, 1, CUDA.Mem.DeviceBuffer},
                    x::CUDA.CuArray{Float64, 1, CUDA.Mem.DeviceBuffer},
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

