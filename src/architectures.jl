
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
