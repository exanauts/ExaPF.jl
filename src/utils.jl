
# norm
xnorm(x::AbstractVector) = norm(x, 2)

xnorm_inf(a) = maximum(abs.(a))

default_sparse_matrix(::CPU) = SparseMatrixCSC{Float64,Int}

function get_jacobian_types(::CPU)
    SMT = SparseMatrixCSC{Float64,Int}
    A = Vector
    return SMT, A
end

#=
    Kernels utils
=#

@kernel function _transfer_to_input!(input, map, src)
    i = @index(Global, Linear)
    input[map[i]] = src[i]
end

@kernel function _transfer_fr_input!(dest, input, map)
    i = @index(Global, Linear)
    dest[i] = input[map[i]]
end

# Differentiable LinearAlgebra.mul! for ForwardDiff
@kernel function _spmv_csr_kernel!(Y, X, colVal, rowPtr, nzVal, alpha, beta, n, m)
    i, k = @index(Global, NTuple)
    Y[k, i] *= beta
    @inbounds for c in rowPtr[i]:rowPtr[i+1]-1
        j = colVal[c]
        Y[k, i] += alpha * nzVal[c] * X[k, j]
    end
end

#=
    CSC2CSR
=#

# Taken from
# https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L376
function csr2csc(n, m, Ap, Aj, Ax, Bp, Bi, Bx)
    nnzA = Ap[n+1] - 1
    fill!(Bp, 0)

    for i in 1:nnzA
        Bp[Aj[i]] += 1
    end

    cumsum = 1
    for j in 1:m
        tmp = Bp[j]
        Bp[j] = cumsum
        cumsum += tmp
    end
    Bp[m+1] = nnzA + 1

    for i in 1:n
        for c in Ap[i]:Ap[i+1]-1
            j = Aj[c]
            dest = Bp[j]
            Bi[dest] = i
            Bx[dest] = Ax[c]
            Bp[j] += 1
        end
    end

    last = 1
    for j in 1:m+1
        tmp = Bp[j]
        Bp[j] = last
        last = tmp
    end
end

csc2csr(n, m, Ap, Ai, Ax, Bp, Bj, Bx) = csr2csc(m, n, Ap, Ai, Ax, Bp, Bj, Bx)

function convert2csr(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    n, m = size(A)
    nnzA = nnz(A)
    Ap, Ai, Ax = A.colptr, A.rowval, A.nzval

    Bp = zeros(Ti, n+1)
    Bj = zeros(Ti, nnzA)
    Bx = zeros(Tv, nnzA)

    csc2csr(n, m, Ap, Ai, Ax, Bp, Bj, Bx)
    return Bp, Bj, Bx
end

