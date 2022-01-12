
import CUDA.CUBLAS
import CUDA.CUSPARSE
import CUDA.CUSOLVER

using CUDAKernels

function PolarForm(pf::PS.PowerNetwork, device::CUDADevice)
    return PolarForm{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}(pf, device)
end

default_sparse_matrix(::CUDADevice) = CuSparseMatrixCSR{Float64, Int}
xnorm(x::CUDA.CuVector) = CUBLAS.nrm2(x)

function get_jacobian_types(::CUDADevice)
    SMT = default_sparse_matrix(CUDADevice())
    A = CUDA.CuVector
    return SMT, A
end

function Base.unsafe_wrap(Atype::Type{CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer}},
                          p::CUDA.CuPtr{T}, dim::Integer;
                          own::Bool=false, ctx::CUDA.CuContext=CUDA.context()) where {T}
    unsafe_wrap(CUDA.CuArray{T, 1}, p, (dim,); own, ctx)
end

# Differentiable LinearAlgebra.mul! for ForwardDiff
@kernel function _spmm_kernel!(Y, X, colVal, rowPtr, nzVal, alpha, beta, n, m)
    i, k = @index(Global, NTuple)
    Y[i, k] *= beta
    @inbounds for c in rowPtr[i]:rowPtr[i+1]-1
        j = colVal[c]
        Y[i, k] += alpha * nzVal[c] * X[j, k]
    end
end

function LinearAlgebra.mul!(Y::AbstractArray{T, 2}, A::CUSPARSE.CuSparseMatrixCSR, X::AbstractArray{T, 2}, alpha::Number, beta::Number) where {T <: ForwardDiff.Dual}
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
