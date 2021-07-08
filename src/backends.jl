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

default_sparse_matrix(::CPU) = SparseMatrixCSC

function get_jacobian_types(device::KA.Device)
    SMT = sparse_matrix_type(getbackend(device)){Float64}
    A = array_type(getbackend(device))
    return SMT, A
end

function get_batch_jacobian_types(device::KA.Device)
    getbackend(device)
    SMT = sparse_matrix_type(getbackend(device)){Float64}
    A = array_type(device)
    return SMT, A
end


# norm
xnorm(x::AbstractVector) = norm(x, 2)
xnorm(x::CUDA.CuVector) = CUBLAS.nrm2(x)
function xnorm(x::AMDGPU.ROCVector)
    x = convert(Vector, x)
    return norm(x, 2)
end

# Array initialization
xzeros(S, n) = fill!(S(undef, n), zero(eltype(S)))
xones(S, n) = fill!(S(undef, n), one(eltype(S)))

xnorm_inf(a) = maximum(abs.(a))

function xnorm_inf(x::AMDGPU.ROCVector)
    x = convert(Vector, x)
    return maximum(abs.(x))
end

function skip_roc(device::KA.Device)
    return getbackend(device) != ROCBackend()
end

export skip_roc, getbackend, HostBackend, CUDABackend, ROCBackend, OneAPIBackend

