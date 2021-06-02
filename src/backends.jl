# using ROCKernels
abstract type AbstractBackend end
# Hardware or software type
struct HostBackend <: AbstractBackend end
struct CUDABackend <: AbstractBackend end
struct ROCBackend <: AbstractBackend end
struct OneAPIBackend <: AbstractBackend end

array_type(::ExaPF.HostBackend) = Array
array_type(::ExaPF.CUDABackend) = CUDA.CuArray
array_type(::ExaPF.ROCBackend) = AMDGPU.ROCArray

vector_type(::ExaPF.HostBackend) = Vector
vector_type(::ExaPF.CUDABackend) = CUDA.CuVector
vector_type(::ExaPF.ROCBackend) = AMDGPU.ROCVector

matrix_type(::ExaPF.HostBackend) = Matrix
matrix_type(::ExaPF.CUDABackend) = CUDA.CuMatrix
matrix_type(::ExaPF.ROCBackend)  = AMDGPU.ROCMatrix

sparse_matrix_type(::ExaPF.HostBackend) = SparseMatrixCSC
sparse_matrix_type(::ExaPF.CUDABackend) = CUDA.CUSPARSE.CuSparseMatrixCSR
sparse_matrix_type(::ExaPF.ROCBackend)  = AMDGPU.ROCMatrix

getbackend(::KA.CPU) = HostBackend()
if CUDA.has_cuda_gpu() 
    using CUDAKernels
    getbackend(::CUDAKernels.CUDADevice) = CUDABackend()
end
if AMDGPU.hsa_configured
    using ROCKernels
    getbackend(::ROCKernels.ROCDevice) = ROCBackend()
end


# norm
xnorm(x::AbstractVector) = norm(x, 2)
xnorm(x::CUDA.CuVector) = CUBLAS.nrm2(x)
xnorm(x::AMDGPU.ROCVector) = 0.0

# Array initialization
xzeros(S, n) = fill!(S(undef, n), zero(eltype(S)))
xones(S, n) = fill!(S(undef, n), one(eltype(S)))

xnorm_inf(a) = maximum(abs.(a))
export getbackend, HostBackend, CUDABackend, ROCBackend, OneAPIBackend

