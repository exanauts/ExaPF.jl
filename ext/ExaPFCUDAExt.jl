module ExaPFCUDAExt

using ExaPF
using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using CUDA.CUSPARSE

using ForwardDiff
using LinearAlgebra
using KernelAbstractions
using SparseArrays

const KA = KernelAbstractions
const LS = ExaPF.LinearSolvers
const PS = ExaPF.PowerSystem
const AD = ExaPF.AutoDiff

LS.DirectSolver(J::CuSparseMatrixCSR; options...) = ExaPF.LS.DirectSolver(nothing)
LS.update!(solver::ExaPF.LS.AbstractIterativeLinearSolver, J::CuSparseMatrixCSR) = ExaPF.LS.update(solver.precond, J, CUDABackend())
LS._get_type(J::CuSparseMatrixCSR) = CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
LS.default_linear_solver(A::CuSparseMatrixCSR, device::CUDABackend) = ExaPF.LS.DirectSolver(A)
LS._allowscalar(f::Function, J::CuSparseMatrixCSR) = CUDA.allowscalar(f)
ExaPF._iscsr(::CuSparseMatrixCSR) = true
ExaPF._iscsc(::CuSparseMatrixCSR) = false
"""
    list_solvers(::CUDABackend)

List all linear solvers available solving the power flow on an NVIDIA GPU.
"""
ExaPF.list_solvers(::CUDABackend) = [LS.DirectSolver, LS.BICGSTAB, LS.DQGMRES, LS.EigenBICGSTAB, LS.KrylovBICGSTAB]

include("cuda_wrapper.jl")
include("cuda_preconditioner.jl")
end
