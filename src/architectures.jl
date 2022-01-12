
abstract type AbstractArchitecture end

# norm
xnorm(x::AbstractVector) = norm(x, 2)
xnorm(x::CUDA.CuVector) = CUBLAS.nrm2(x)

xnorm_inf(a) = maximum(abs.(a))

default_sparse_matrix(::CPU) = SparseMatrixCSC{Float64,Int}

function get_jacobian_types(::CPU)
    SMT = SparseMatrixCSC{Float64,Int}
    A = Vector
    return SMT, A
end

function get_jacobian_types(::GPU)
    SMT = CUSPARSE.CuSparseMatrixCSR{Float64}
    A = CUDA.CuVector
    return SMT, A
end

function get_batch_jacobian_types(::CPU)
    SMT = SparseMatrixCSC{Float64,Int}
    A = Array
    return SMT, A
end

function get_batch_jacobian_types(::GPU)
    SMT = CUSPARSE.CuSparseMatrixCSR{Float64}
    A = CUDA.CuArray
    return SMT, A
end

function Base.unsafe_wrap(Atype::Type{CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer}},
                          p::CUDA.CuPtr{T}, dim::Integer;
                          own::Bool=false, ctx::CUDA.CuContext=CUDA.context()) where {T}
    unsafe_wrap(CUDA.CuArray{T, 1}, p, (dim,); own, ctx)
end

