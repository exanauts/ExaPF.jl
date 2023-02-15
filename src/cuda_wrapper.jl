
using CUDA
import CUDA.CUBLAS
import CUDA.CUSPARSE: CuSparseMatrixCSR, CuSparseMatrixCSC
import CUDA.CUSOLVER

function PolarForm(pf::PS.PowerNetwork, device::CUDABackend, ncustoms::Int=0)
    return PolarForm{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}(pf, device, ncustoms)
end
function BlockPolarForm(pf::PS.PowerNetwork, device::CUDABackend, k::Int, ncustoms::Int=0)
    return BlockPolarForm{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}(pf, device, k, ncustoms)
end
function PolarFormRecourse(pf::PS.PowerNetwork, device::CUDABackend, k::Int)
    ngen = PS.get(pf, PS.NumberOfGenerators())
    ncustoms = (ngen + 1) * k
    return PolarFormRecourse{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}(pf, device, k, ncustoms)
end

default_sparse_matrix(::CUDABackend) = CuSparseMatrixCSR{Float64, Int32}
xnorm(x::CUDA.CuVector) = CUBLAS.nrm2(x)

function get_jacobian_types(::CUDABackend)
    SMT = CuSparseMatrixCSR{Float64, Int32}
    A = CUDA.CuVector
    return SMT, A
end

function Base.unsafe_wrap(Atype::Type{CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer}},
                          p::CUDA.CuPtr{T}, dim::Integer;
                          own::Bool=false, ctx::CUDA.CuContext=CUDA.context()) where {T}
    unsafe_wrap(CUDA.CuArray{T, 1}, p, (dim,); own, ctx)
end

CuSparseMatrixCSR{Tv, Int32}(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti} = CuSparseMatrixCSR(A)


# AbstractStack

function Base.copyto!(stack::AutoDiff.AbstractStack, map::AbstractVector{Int}, vals::VT) where {VT <: CuArray}
    @assert length(map) == length(vals)
    ndrange = (length(map),)
    _transfer_to_input!(CUDABackend())(
        stack.input, map, vals;
        ndrange=ndrange,
    )
    KA.synchronize(CUDABackend())
end

function Base.copyto!(dest::VT, stack::AutoDiff.AbstractStack, map::AbstractVector{Int}) where {VT <: CuArray}
    @assert length(map) == length(dest)
    ndrange = (length(map),)
    _transfer_fr_input!(CUDABackend())(
        dest, stack.input, map;
        ndrange=ndrange,
    )
    KA.synchronize(CUDABackend())
end

# By default, no factorization routine is available
LinearSolvers.update!(s::DirectSolver{Nothing}, J::CuSparseMatrixCSR) = nothing
function LinearSolvers.ldiv!(::DirectSolver{Nothing},
    y::CuVector, J::CuSparseMatrixCSR, x::CuVector,
)
    CUSOLVER.csrlsvqr!(J, x, y, 1e-8, one(Cint), 'O')
    return 0
end

#=
    Generic SpMV for CuSparseMatrixCSR
=#
function ForwardDiff.npartials(vec::CuArray{ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    return N
end

function _tranpose_descriptor(x::DenseCuMatrix)
    desc_ref = Ref{CUSPARSE.cusparseDnMatDescr_t}()
    n, m = size(x)
    CUSPARSE.cusparseCreateDnMat(desc_ref, n, m, m, x, eltype(x), CUSPARSE.CUSPARSE_ORDER_ROW)
    return desc_ref[]
end

function _mm_transposed!(
    transa::CUSPARSE.SparseChar, transb::CUSPARSE.SparseChar,
    alpha::Number, A::CuSparseMatrixCSR{T},
    B::DenseCuMatrix{T}, beta::Number, C::DenseCuMatrix{T},
    index::CUSPARSE.SparseChar, algo=CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT,
) where {T}
    m,k = size(A)
    n = size(C)[2]

    descA = CUSPARSE.CuSparseMatrixDescriptor(A, index)
    descB = _tranpose_descriptor(B)
    descC = _tranpose_descriptor(C)

    function bufferSize()
        out = Ref{Csize_t}()
        CUSPARSE.cusparseSpMM_bufferSize(
            CUSPARSE.handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, out)
        return out[]
    end
    CUSPARSE.with_workspace(bufferSize) do buffer
        CUSPARSE.cusparseSpMM(
            CUSPARSE.handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, buffer)
    end

    return C
end

function LinearAlgebra.mul!(
    Y::CuArray{T, 1},
    A::CUSPARSE.CuSparseMatrixCSR,
    X::CuArray{T, 1},
    alpha::Number, beta::Number,
) where {T <: ForwardDiff.Dual}
    n, m = size(A)
    @assert size(Y, 1) == n
    @assert size(X, 1) == m

    N = ForwardDiff.npartials(Y)
    p = 1 + N

    # Reinterpret duals as double.
    Xs = reshape(reinterpret(Float64, X), m, p)
    Ys = reshape(reinterpret(Float64, Y), n, p)

    _mm_transposed!('N', 'N', alpha, A, Xs, beta, Ys, 'O')
end

function LinearAlgebra.mul!(
    Y::CuArray{T, 1},
    A::CUSPARSE.CuSparseMatrixCSR,
    X::AbstractArray{Float64, 1},
    alpha::Number, beta::Number,
) where {T <: ForwardDiff.Dual}
    n, m = size(A)
    @assert size(Y, 1) == n
    @assert size(X, 1) == m

    N = ForwardDiff.npartials(Y)
    p = 1 + N

    # Reinterpret duals as double.
    Ys = reshape(reinterpret(Float64, Y), p, n)

    ndrange = (n, )
    _spmv_csr_kernel_double!(CUDABackend())(
        Ys, X, A.colVal, A.rowPtr, A.nzVal, alpha, beta, n, m;
        ndrange=ndrange,
    )
    KA.synchronize(CUDABackend())
end

#=
    Generic SpMV for CuSparseMatrixCSC
=#

function LinearAlgebra.mul!(
    Y::AbstractArray{Td, 1},
    A::Adjoint{T, CuSparseMatrixCSR{T, I}},
    X::AbstractArray{Td, 1},
    alpha::Number, beta::Number,
) where {I, T, Td <: ForwardDiff.Dual}
    n, m = size(A)
    p = ForwardDiff.npartials(Y) + 1
    @assert size(Y, 1) == n
    @assert size(X, 1) == m

    B = A.parent

    nthreads = 256
    threads_y = p
    threads_x = div(nthreads, threads_y)
    threads = (threads_x, threads_y)

    blocks = ceil.(Int, (m, p) ./ threads)

    # Reinterpret duals as double.
    Ys = reshape(reinterpret(Float64, Y), n, p)
    Xs = reshape(reinterpret(Float64, X), m, p)
    _mm_transposed!('T', 'N', alpha, A.parent, Xs, beta, Ys, 'O')
end

@kernel function _blk_transfer_to_input!(input, map, src, nx)
    i, k = @index(Global, NTuple)
    input[map[i + (k-1)*nx]] = src[i]
end

function blockcopy!(stack::NetworkStack, map::CuArray{Int}, x::CuArray{Float64})
    nx = length(x)
    @assert length(map) % nx == 0
    nb = div(length(map), nx)
    ndrange = (nx, nb)
    _blk_transfer_to_input!(CUDABackend())(
        stack.input, map, x, nx;
        ndrange=ndrange,
    )
    KA.synchronize(CUDABackend())
end

