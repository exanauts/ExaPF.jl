module ExaPFCUDAExt

using ExaPF
using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using CUDA.CUSPARSE
using CUDSS
using GPUArraysCore

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

import ..ExaPF.AD: AbstractJacobian

function LS.DirectSolver(A::AbstractJacobian, ::CUDABackend, nblocks=1)
    J = A.J
    cudss_solver = if nblocks == 1
        lu(J)
    elseif nblocks > 1
        n, rem = divrem(size(J, 1), nblocks)
        @assert rem == 0 "Number of rows must be divisible by number of blocks"
        rowPtr    = J.rowPtr[1:n+1]
        @allowscalar nnz_start = J.rowPtr[1]
        @allowscalar nnz_end   = J.rowPtr[n+1] - 1    # last non-zero index for first block
        colVal    = J.colVal[nnz_start:nnz_end]

        lu(
            CuSparseMatrixCSR(
                rowPtr,
                colVal,
                J.nzVal,
                (n,n),
            )
        )
    else
        error("Number of blocks must be >= 1")
    end
    ds = LS.DirectSolver(cudss_solver)
    return ds
end

LS.update!(is::ExaPF.LS.AbstractIterativeLinearSolver, J::CuSparseMatrixCSR) = KP.update!(is.precond, J)
LS.update!(ds::ExaPF.LS.DirectSolver, J::CuSparseMatrixCSR) = lu!(ds.factorization, J)
LS._get_type(J::CuSparseMatrixCSR) = CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
function LS.default_linear_solver(A::ExaPF.BatchJacobian, backend::CUDABackend, nblocks=1)
    return ExaPF.LS.DirectSolver(A, backend, A.nblocks)
end

ExaPF._iscsr(::CuSparseMatrixCSR) = true
ExaPF._iscsc(::CuSparseMatrixCSR) = false
function LS.scaling!(::LS.Bicgstab, A::CuSparseMatrixCSR, b)
    KP.scaling_csr!(A,b)
end

"""
    list_solvers(::CUDABackend)

List all linear solvers available for solving the (batch) power flow on an NVIDIA GPU.
"""
ExaPF.list_solvers(::CUDABackend) = [LS.DirectSolver, LS.Dqgmres, LS.Bicgstab]

include("cuda_wrapper.jl")

end
