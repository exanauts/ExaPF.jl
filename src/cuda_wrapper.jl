
using CUDA
import CUDA.CUBLAS
import CUDA.CUSPARSE: CuSparseMatrixCSR, CuSparseMatrixCSC
import CUDA.CUSOLVER

using CUDAKernels

export CUDAKernels

function PolarForm(pf::PS.PowerNetwork, device::CUDADevice)
    return PolarForm{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}(pf, device)
end

default_sparse_matrix(::CUDADevice) = CuSparseMatrixCSR{Float64, Int32}
xnorm(x::CUDA.CuVector) = CUBLAS.nrm2(x)

function get_jacobian_types(::CUDADevice)
    SMT = CuSparseMatrixCSR{Float64, Int32}
    A = CUDA.CuVector
    return SMT, A
end

function Base.unsafe_wrap(Atype::Type{CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer}},
                          p::CUDA.CuPtr{T}, dim::Integer;
                          own::Bool=false, ctx::CUDA.CuContext=CUDA.context()) where {T}
    unsafe_wrap(CUDA.CuArray{T, 1}, p, (dim,); own, ctx)
end

CuSparseMatrixCSR{Tv, Int32}(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti} = CuSparseMatrixCSR(A)


# AbstractStack

@kernel function _transfer_to_input!(input, map, src)
    i = @index(Global, Linear)
    input[map[i]] = src[i]
end

@kernel function _transfer_fr_input!(dest, input, map)
    i = @index(Global, Linear)
    dest[i] = input[map[i]]
end

function Base.copyto!(stack::AutoDiff.AbstractStack, map::AbstractVector{Int}, vals::VT) where {VT <: CuArray}
    @assert length(map) == length(vals)
    ndrange = (length(map),)
    ev = _transfer_to_input!(CUDADevice())(stack.input, map, vals, ndrange=ndrange)
    wait(ev)
end

function Base.copyto!(dest::VT, stack::AutoDiff.AbstractStack, map::AbstractVector{Int}) where {VT <: CuArray}
    @assert length(map) == length(dest)
    ndrange = (length(map),)
    ev = _transfer_fr_input!(CUDADevice())(dest, stack.input, map, ndrange=ndrange)
    wait(ev)
end

#=
    LinearSolvers
=#
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

# By default, no factorization routine is available
LinearSolvers.update!(s::DirectSolver{Nothing}, J::CuSparseMatrixCSR) = nothing
function LinearSolvers.ldiv!(::DirectSolver{Nothing},
    y::CuVector, J::CuSparseMatrixCSR, x::CuVector,
)
    CUSOLVER.csrlsvqr!(J, x, y, 1e-8, one(Cint), 'O')
    return 0
end
function LinearSolvers.ldiv!(::DirectSolver{Nothing},
    y::CUDA.CuVector, J::CUSPARSE.CuSparseMatrixCSC, x::CUDA.CuVector,
)
    csclsvqr!(J, x, y, 1e-8, one(Cint), 'O')
    return 0
end

#=
    Autodiff
=#

@kernel function _extract_values_kernel(dest, src)
    i = @index(Global, Linear)
    dest[i] = src[i].value
end

function extract_values!(dest::CuArray, src::CuArray)
    ndrange = (length(dest),)
    ev = _extract_values_kernel(CUDADevice())(dest, src, ndrange=ndrange)
    wait(ev)
end

#=
    Generic SpMV for CuSparseMatrixCSR
=#
function ForwardDiff.npartials(vec::CuVector{ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    return N
end

# Differentiable LinearAlgebra.mul! for ForwardDiff
@kernel function _spmv_csr_kernel!(Y, X, colVal, rowPtr, nzVal, alpha, beta, n, m)
    i, k = @index(Global, NTuple)
    Y[k, i] *= beta
    @inbounds for c in rowPtr[i]:rowPtr[i+1]-1
        j = colVal[c]
        Y[k, i] += alpha * nzVal[c] * X[k, j]
    end
end

function LinearAlgebra.mul!(
    Y::CuArray{T, 1},
    A::CUSPARSE.CuSparseMatrixCSR,
    X::AbstractArray{T, 1},
    alpha::Number, beta::Number,
) where {T <: ForwardDiff.Dual}
    n, m = size(A)
    @assert size(Y, 1) == n
    @assert size(X, 1) == m

    N = ForwardDiff.npartials(Y)
    p = 1 + N

    # Reinterpret duals as double.
    Ys = reshape(reinterpret(Float64, Y), p, n)
    Xs = reshape(reinterpret(Float64, X), p, m)

    ndrange = (n, p)
    ev = _spmv_csr_kernel!(CUDADevice())(
        Ys, Xs, A.colVal, A.rowPtr, A.nzVal, alpha, beta, n, m,
        ndrange=ndrange, dependencies=Event(CUDADevice()),
    )
    wait(ev)
end


#=
    Generic SpMV for CuSparseMatrixCSC
=#

# Write a CUDA kernel directly as KernelAbstractions does not
# supports atomic_add.
function _spmm_csc_kernel_T!(Y, X, colPtr, rowVal, nzVal, alpha, beta, m, p)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    J = threadIdx().y + (blockDim().y * (blockIdx().y - 1))
    if I <= m && J <= p
        @inbounds for c in colPtr[I]:colPtr[I+1]-1
            j = rowVal[c]
            CUDA.@atomic Y[J, j] += alpha * nzVal[c] * X[J, I]
        end
    end
end

function LinearAlgebra.mul!(
    Y::AbstractArray{D, 1},
    A::Adjoint{T, CuSparseMatrixCSR{T, I}},
    X::AbstractArray{D, 1},
    alpha::Number, beta::Number,
) where {N, I, T, S, D <: ForwardDiff.Dual{S, T, N}}
    n, m = size(A)
    p = N + 1
    @assert size(Y, 1) == n
    @assert size(X, 1) == m

    B = A.parent

    nthreads = 256
    threads_y = p
    threads_x = div(nthreads, threads_y)
    threads = (threads_x, threads_y)

    blocks = ceil.(Int, (m, p) ./ threads)

    # Reinterpret duals as double.
    # (Needed to work with atomic_add)
    Ys = reshape(reinterpret(Float64, Y), p, n)
    Xs = reshape(reinterpret(Float64, X), p, m)

    Ys .*= beta
    @cuda threads=threads blocks=blocks _spmm_csc_kernel_T!(
        Ys, Xs, B.rowPtr, B.colVal, B.nzVal, alpha, beta, m, p,
    )
end

