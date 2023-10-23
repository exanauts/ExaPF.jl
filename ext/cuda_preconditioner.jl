LS.BlockJacobiPreconditioner(J::CUSPARSE.CuSparseMatrixCSR; options...) = ExaPF.LS.BlockJacobiPreconditioner(SparseMatrixCSC(J); options...)

function _update_gpu(p, j_rowptr, j_colval, j_nzval, device::CUDABackend)
    nblocks = p.nblocks
    fillblock_gpu_kernel! = ExaPF.LS._fillblock_gpu!(device)
    # Fill Block Jacobi" begin
    fillblock_gpu_kernel!(
        p.cublocks, size(p.id,1),
        p.cupartitions, p.cumap,
        j_rowptr, j_colval, j_nzval,
        p.cupart, p.culpartitions, p.id,
        ndrange=nblocks,
    )
    KA.synchronize(device)
    # Invert blocks begin
    blocklist = Array{CuArray{Float64,2}}(undef, nblocks)
    for b in 1:nblocks
        blocklist[b] = p.cublocks[:,:,b]
    end
    CUDA.@sync pivot, info = CUDA.CUBLAS.getrf_batched!(blocklist, true)
    CUDA.@sync pivot, info, blocklist = CUDA.CUBLAS.getri_batched(blocklist, pivot)
    for b in 1:nblocks
        p.cublocks[:,:,b] .= blocklist[b]
    end
    return
end

"""
    function update(J::CuSparseMatrixCSR, p)

Update the preconditioner `p` from the sparse Jacobian `J` in CSR format for the GPU

1) The dense blocks `cuJs` are filled from the sparse Jacobian `J`
2) To a batch inversion of the dense blocks using CUBLAS
3) Extract the preconditioner matrix `p.P` from the dense blocks `cuJs`

"""
function LS.update(p, J::CUSPARSE.CuSparseMatrixCSR, device::CUDABackend)
    _update_gpu(p, J.rowPtr, J.colVal, J.nzVal, device)
end
