
abstract type AbstractArchitecture end

# norm
xnorm(x::AbstractVector) = norm(x, 2)
xnorm(x::CUDA.CuVector) = CUBLAS.nrm2(x)

xnorm_inf(a) = maximum(abs.(a))

default_sparse_matrix(::CPU) = SparseMatrixCSC
