
using CUDAKernels
using BlockPowerFlow

using CUDA.CUSPARSE
import ExaPF: LinearSolvers
import BlockPowerFlow: CUSOLVERRF

const LS = LinearSolvers
const CUDA_ARCH = (CUDADevice(), CuArray, CuSparseMatrixCSR)

# Overload factorization routine to use cusolverRF
LS.exa_factorize(J::CuSparseMatrixCSR) = CUSOLVERRF.CusolverRfLU(J)
LS.exa_factorize(J::CuSparseMatrixCSC) = CUSOLVERRF.CusolverRfLU(J)

# Overload factorization for batch Hessian computation
function ExaPF._batch_hessian_factorization(J::CuSparseMatrixCSR, nbatch)
    Jtrans = CUSPARSE.CuSparseMatrixCSC(J)
    if nbatch == 1
        lufac = CUSOLVERRF.CusolverRfLU(J)
        lufact = CUSOLVERRF.CusolverRfLU(Jtrans)
    else
        lufac = CUSOLVERRF.CusolverRfLUBatch(J, nbatch)
        lufact = CUSOLVERRF.CusolverRfLUBatch(Jtrans, nbatch)
    end
    return (lufac, lufact)
end

