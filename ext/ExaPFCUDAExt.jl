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

LS.DirectSolver(J::CuSparseMatrixCSR; options...) = ExaPF.LS.DirectSolver(lu(J))
LS.update!(solver::ExaPF.LS.AbstractIterativeLinearSolver, J::CuSparseMatrixCSR) = KP.update!(solver.precond, J)
LS.update!(solver::ExaPF.LS.DirectSolver, J::CuSparseMatrixCSR) = lu!(solver.factorization, J)
LS._get_type(J::CuSparseMatrixCSR) = CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
LS.default_linear_solver(A::CuSparseMatrixCSR, device::CUDABackend) = ExaPF.LS.DirectSolver(A)
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
