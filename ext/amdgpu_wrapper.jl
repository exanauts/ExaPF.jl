function ExaPF.PolarForm(pf::PS.PowerNetwork, device::ROCBackend, ncustoms::Int=0)
    return PolarForm{Float64, ROCVector{Int}, ROCVector{Float64}, ROCMatrix{Float64}}(pf, device, ncustoms)
end
function ExaPF.BlockPolarForm(pf::PS.PowerNetwork, device::ROCBackend, k::Int, ncustoms::Int=0)
    return BlockPolarForm{Float64, ROCVector{Int}, ROCVector{Float64}, ROCMatrix{Float64}}(pf, device, k, ncustoms)
end
function ExaPF.PolarFormRecourse(pf::PS.PowerNetwork, device::ROCBackend, k::Int)
    ngen = PS.get(pf, PS.NumberOfGenerators())
    ncustoms = (ngen + 1) * k
    return PolarFormRecourse{Float64, ROCVector{Int}, ROCVector{Float64}, ROCMatrix{Float64}}(pf, device, k, ncustoms)
end

ExaPF.default_sparse_matrix(::ROCBackend) = ROCSparseMatrixCSR{Float64, Int32}
ExaPF.xnorm(x::AMDGPU.ROCVector) = rocBLAS.nrm2(length(x), x, 1)

function ExaPF.get_jacobian_types(::ROCBackend)
    SMT = ROCSparseMatrixCSR{Float64, Int32}
    A = AMDGPU.ROCVector
    return SMT, A
end

function Base.unsafe_wrap(Atype::Type{AMDGPU.ROCArray{T, 1}},
                          p::AMDGPU.Ptr{T}, dim::Integer;
                          own::Bool=false) where {T}
    unsafe_wrap(AMDGPU.ROCVector{T}, p, (dim,); own)
end

rocSPARSE.ROCSparseMatrixCSR{Tv, Int32}(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti} = ROCSparseMatrixCSR(A)


# AbstractStack

function Base.copyto!(stack::AD.AbstractStack, map::AbstractVector{Int}, vals::VT) where {VT <: ROCArray}
    @assert length(map) == length(vals)
    ndrange = (length(map),)
    ExaPF._transfer_to_input!(ROCBackend())(
        stack.input, map, vals;
        ndrange=ndrange,
    )
    KA.synchronize(ROCBackend())
end

function Base.copyto!(dest::VT, stack::AD.AbstractStack, map::AbstractVector{Int}) where {VT <: ROCArray}
    @assert length(map) == length(dest)
    ndrange = (length(map),)
    ExaPF._transfer_fr_input!(ROCBackend())(
        dest, stack.input, map;
        ndrange=ndrange,
    )
    KA.synchronize(ROCBackend())
end

# By default, no factorization routine is available
LS.update!(s::LS.DirectSolver{Nothing}, J::ROCSparseMatrixCSR) = nothing

#=
    Generic SpMV for ROCSparseMatrixCSR
=#
function ExaPF.ForwardDiff.npartials(vec::ROCArray{ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    return N
end

function _transpose_descriptor(x::DenseROCMatrix)
    desc_ref = Ref{rocSPARSE.rocsparse_dnmat_descr}()
    n, m = size(x)
    rocSPARSE.rocsparse_create_dnmat_descr(desc_ref, n, m, m, x, eltype(x), rocSPARSE.rocsparse_order_row)
    return desc_ref[]
end

function _mm_transposed!(
    transa::rocSPARSE.SparseChar, transb::rocSPARSE.SparseChar,
    alpha::Number, A::ROCSparseMatrixCSR{T},
    B::DenseROCMatrix{T}, beta::Number, C::DenseROCMatrix{T},
    index::rocSPARSE.SparseChar, algo=rocSPARSE.rocsparse_spmm_alg_default,
) where {T}
    m,k = size(A)
    n = size(C)[2]

    descA = rocSPARSE.ROCSparseMatrixDescriptor(A, index)
    descB = _transpose_descriptor(B)
    descC = _transpose_descriptor(C)

    function bufferSize()
        out = Ref{Csize_t}()
        rocSPARSE.rocsparse_spmm(
            rocSPARSE.handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, rocSPARSE.rocsparse_spmm_stage_buffer_size, out, C_NULL)
        return out[]
    end
    rocSPARSE.with_workspace(bufferSize) do buffer
        buffer_len_ref = Ref{Csize_t}(sizeof(buffer))
        rocSPARSE.rocsparse_spmm(
            rocSPARSE.handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, rocSPARSE.rocsparse_spmm_stage_preprocess, buffer_len_ref, buffer)
        rocSPARSE.rocsparse_spmm(
            rocSPARSE.handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
            descC, T, algo, rocSPARSE.rocsparse_spmm_stage_compute, buffer_len_ref, buffer)
    end

    return C
end

function LinearAlgebra.mul!(
    Y::ROCArray{T, 1},
    A::rocSPARSE.ROCSparseMatrixCSR,
    X::ROCArray{T, 1},
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
    # mm!('N', 'N', alpha, A, Xs, beta, Ys, 'O', rocSPARSE.rocsparse_spmm_alg_default)
    _mm_transposed!('N', 'N', alpha, A, Xs, beta, Ys, 'O')
end

function LinearAlgebra.mul!(
    Y::ROCArray{T, 1},
    A::rocSPARSE.ROCSparseMatrixCSR,
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
    ExaPF._spmv_csr_kernel_double!(ROCBackend())(
        Ys, X, A.colVal, A.rowPtr, A.nzVal, alpha, beta, n, m;
        ndrange=ndrange,
    )
    KA.synchronize(ROCBackend())
end

#=
    Generic SpMV for CuSparseMatrixCSC
=#

function LinearAlgebra.mul!(
    Y::AbstractArray{Td, 1},
    A::Adjoint{T, ROCSparseMatrixCSR{T, I}},
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
    # _mm_transposed!('T', 'N', alpha, A.parent, Xs, beta, Ys, 'O')
    mm!('T', 'N', alpha, A.parent, Xs, beta, Ys, 'O', rocSPARSE.rocsparse_spmm_alg_default)

end

function ExaPF.blockcopy!(stack::ExaPF.NetworkStack, map::ROCArray{Int}, x::ROCArray{Float64})
    nx = length(x)
    @assert length(map) % nx == 0
    nb = div(length(map), nx)
    ndrange = (nx, nb)
    ExaPF._blk_transfer_to_input!(ROCBackend())(
        stack.input, map, x, nx;
        ndrange=ndrange,
    )
    KA.synchronize(ROCBackend())
end
