module ExaPFAMDGPUExt

using ExaPF
using AMDGPU
using AMDGPU.rocBLAS
using AMDGPU.rocSOLVER
using AMDGPU.rocSPARSE

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

LS.DirectSolver(J::ROCSparseMatrixCSR; options...) = ExaPF.LS.DirectSolver(nothing)
LS.update!(solver::ExaPF.LS.AbstractIterativeLinearSolver, J::ROCSparseMatrixCSR) = KP.update!(solver.precond, J)
LS._get_type(J::ROCSparseMatrixCSR) = ROCArray{Float64, 1, AMDGPU.Mem.HIPBuffer}
LS.default_linear_solver(A::ROCSparseMatrixCSR, device::ROCBackend) = ExaPF.LS.KrylovBICGSTAB(A)
ExaPF._iscsr(::ROCSparseMatrixCSR) = true
ExaPF._iscsc(::ROCSparseMatrixCSR) = false
function LS.scaling!(::LS.KrylovBICGSTAB,A::ROCSparseMatrixCSR,b)
    KP.scaling_csr!(A,b)
end

"""
    list_solvers(::ROCBackend)

List all linear solvers available solving the power flow on an NVIDIA GPU.
"""
ExaPF.list_solvers(::ROCBackend) = [LS.BICGSTAB, LS.DQGMRES, LS.EigenBICGSTAB, LS.KrylovBICGSTAB]

include("amdgpu_wrapper.jl")
end
