module ExaPFCUDAExt

using ExaPF
using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using CUDA.CUSPARSE
using CUDSS

using ForwardDiff
using LinearAlgebra
using KernelAbstractions
using SparseArrays
using KrylovPreconditioners

const KA = KernelAbstractions
const LS = ExaPF.LinearSolvers
const PS = ExaPF.PowerSystem
const AD = ExaPF.AutoDiff
const KP = KrylovPreconditioners

function LS.DirectSolver(J::CuSparseMatrixCSR, nbatch::Int=1; options...)
    @assert nbatch ≥ 1
    if nbatch == 1
        cudss_solver = lu(J)
        CUDA.synchronize()
        ds = LS.DirectSolver(cudss_solver)
        return ds
    else
        k = length(J.rowPtr) - 1
        cudss_solver = CudssSolver(J, "G", 'F')
        cudss_set(solver, "ubatch_size", nbatch)
        cudss_b_gpu = CudssMatrix(Float64, k; nbatch)
        cudss_x_gpu = CudssMatrix(Float64, k; nbatch)
        cudss("analysis", cudss_solver, cudss_x_gpu, cudss_b_gpu)
        cudss("factorization", cudss_solver, cudss_x_gpu, cudss_b_gpu)
        CUDA.synchronize()
        ds = LS.DirectSolver(cudss_solver)
        return ds
    end
end

function ldiv!(s::DirectSolver{<:CUDSS.CudssSolver}, y::CuVector, J::AbstractMatrix, x::CuVector; options...)
    LinearAlgebra.ldiv!(y, s.factorization, x)
    CUDA.synchronize()
    return 0
end

function ldiv!(s::DirectSolver{<:CUDSS.CudssSolver}, y::CuArray, x::CuArray; options...)
    LinearAlgebra.ldiv!(y, s.factorization, x)
    CUDA.synchronize()
    return 0
end

function ldiv!(s::DirectSolver{<:CUDSS.CudssSolver}, y::CuArray; options...)
    LinearAlgebra.ldiv!(s.factorization, y)
    CUDA.synchronize()
    return 0
end

LS.update!(solver::ExaPF.LS.AbstractIterativeLinearSolver, J::CuSparseMatrixCSR) = KP.update!(solver.precond, J)
LS.update!(solver::ExaPF.LS.DirectSolver, J::CuSparseMatrixCSR) = lu!(solver.factorization, J); CUDA.synchronize()
LS._get_type(J::CuSparseMatrixCSR) = CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
LS.default_linear_solver(A::CuSparseMatrixCSR, device::CUDABackend) = ExaPF.LS.DirectSolver(A)

function LS.default_batch_linear_solver(A::CuSparseMatrixCSR, device::CUDABackend)
    nbatch = length(A.nzVal) ÷ length(A.colVal)
    ExaPF.LS.DirectSolver(A, nbatch)
end

ExaPF._iscsr(::CuSparseMatrixCSR) = true
ExaPF._iscsc(::CuSparseMatrixCSR) = false
function LS.scaling!(::LS.Bicgstab, A::CuSparseMatrixCSR, b)
    KP.scaling_csr!(A,b)
end

"""
    list_solvers(::CUDABackend)

List all linear solvers available solving the power flow on an NVIDIA GPU.
"""
ExaPF.list_solvers(::CUDABackend) = [LS.DirectSolver, LS.Dqgmres, LS.Bicgstab]

include("cuda_wrapper.jl")
end
