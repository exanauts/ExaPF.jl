# Setup GPU backends dynamically
# This file conditionally loads GPU packages based on availability
using KernelAbstractions
using SparseArrays

is_package_installed(name::String) = !isnothing(Base.find_package(name))

# Try to load CUDA
const CUDA_AVAILABLE = is_package_installed("CUDA")
if CUDA_AVAILABLE
    @eval using CUDA
    @eval using CUDA.CUSPARSE
    CUDA.allowscalar(false)
end

# Try to load AMDGPU
const AMDGPU_AVAILABLE = is_package_installed("AMDGPU")
if AMDGPU_AVAILABLE
    @eval using AMDGPU
    @eval using AMDGPU.rocSPARSE
    AMDGPU.allowscalar(false)
end

# Check functionality
const test_cuda = CUDA_AVAILABLE && CUDA.functional()
const test_rocm = AMDGPU_AVAILABLE && AMDGPU.functional()

# Setup architecture list
const ARCHS = Any[(CPU(), Array, SparseMatrixCSC, "cpu")]

if test_cuda
    CUDA_ARCH = (CUDABackend(), CuArray, CuSparseMatrixCSR, "cuda")
    push!(ARCHS, CUDA_ARCH)
end

if test_rocm
    ROC_ARCH = (ROCBackend(), ROCArray, ROCSparseMatrixCSR, "rocm")
    push!(ARCHS, ROC_ARCH)
end
