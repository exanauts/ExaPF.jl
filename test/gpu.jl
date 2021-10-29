using LinearAlgebra
using CUDAKernels
using CUDA.CUSPARSE

CUDA_ARCH = (CUDADevice(), CuArray, CuSparseMatrixCSR)
push!(ARCHS, CUDA_ARCH)

# Default sparse matrix on CUDA GPU
ExaPF.default_sparse_matrix(::CUDADevice) = CuSparseMatrixCSR

# Differentiable LinearAlgebra.mul! for ForwardDiff
@kernel function _spmm_kernel!(Y, X, colVal, rowPtr, nzVal, alpha, beta, n, m)
    i, k = @index(Global, NTuple)
    Y[i, k] *= beta
    @inbounds for c in rowPtr[i]:rowPtr[i+1]-1
        j = colVal[c]
        Y[i, k] += alpha * nzVal[c] * X[j, k]
    end
end

function LinearAlgebra.mul!(Y::AbstractArray{T, 2}, A::CuSparseMatrixCSR, X::AbstractArray{T, 2}, alpha::Number, beta::Number) where {T <: ForwardDiff.Dual}
    n, m = size(A)
    p = size(X, 2)
    @assert size(Y, 1) == n
    @assert size(X, 1) == m
    @assert size(X, 2) == size(Y, 2)

    ndrange = (n, p)
    ev = _spmm_kernel!(CUDADevice())(
        Y, X, A.colVal, A.rowPtr, A.nzVal, alpha, beta, n, m,
        ndrange=ndrange,
    )
    wait(ev)
end
