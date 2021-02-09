
abstract type AbstractArchitecture end

array_type(::KA.CPU) = Array
array_type(::KA.CUDADevice) = CUDA.CuArray

# norm
xnorm(x::AbstractVector) = norm(x, 2)
xnorm(x::CUDA.CuVector) = CUBLAS.nrm2(x)

# Array initialization
xzeros(S, n) = fill!(S(undef, n), zero(eltype(S)))
xones(S, n) = fill!(S(undef, n), one(eltype(S)))

