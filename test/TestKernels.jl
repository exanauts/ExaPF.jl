# Tests for custom kernels
module TestKernels

using LinearAlgebra
using Random
using SparseArrays
using Test

using ExaPF
using KernelAbstractions
using SparseMatrixColorings
const KA = KernelAbstractions

const AD = ExaPF.AutoDiff

function test_utils_kernel(device, AT, SMT)
    # transfer kernels
    n = 10
    m = 20
    # FROM
    mapping = randperm(m)[1:n]
    dest = zeros(n)
    src = rand(m)
    ndrange = (n, )
    ExaPF._transfer_fr_input!(device)(dest, src, mapping; ndrange=ndrange)
    KA.synchronize(device)
    @test dest == src[mapping]

    # TO
    src = rand(n)
    dest = zeros(m)
    ndrange = (n, )
    ExaPF._transfer_to_input!(device)(dest, mapping, src; ndrange=ndrange)
    KA.synchronize(device)
    @test src == dest[mapping]
    return
end

function test_polar_kernel(device, AT, SMT)
    nb = 10
    nl = 10
    m = nb + 2*nl
    vmag = rand(nb)
    vang = rand(nb)
    output = zeros(m)
    f = rand(1:nb, nl)
    t = rand(1:nb, nl)
    ndrange = (m, 1)
    ExaPF.basis_kernel!(device)(output, vmag, vang, f, t, nl, nb; ndrange=ndrange)
    KA.synchronize(device)

    Δθ = vang[f] .- vang[t]
    vl2 = vmag[f] .* vmag[t]
    res = [vl2 .* cos.(Δθ);  vl2 .* sin.(Δθ); vmag .* vmag]
    @test res == output
    return
end

function test_autodiff_kernel(device, AT, SMT)
    n, m = 10, 20
    J = sprandn(n, m, .2)
    problem = SparseMatrixColorings.ColoringProblem{:nonsymmetric, :column}()
    order = SparseMatrixColorings.NaturalOrder()
    algo = SparseMatrixColorings.GreedyColoringAlgorithm{:direct}(order)
    colors = SparseMatrixColorings.fast_coloring(J, problem, algo; symmetric_pattern=false)
    p = length(unique(colors))

    # set_value!
    x = rand(n)
    duals = zeros(p+1, n)
    ndrange = (n, )
    AD._set_value_kernel!(device)(duals, x; ndrange=ndrange)
    KA.synchronize(device)
    @test duals[1, :] == x

    # Seed with coloring for Jacobian/Hessian
    coloring = rand(1:p, n)
    duals = zeros(p+1, n)
    map = randperm(n)
    ndrange = (n, p)
    AD._seed_coloring_kernel!(device)(duals, coloring, map; ndrange=ndrange)
    KA.synchronize(device)

    # Seed for Hessian-vector products
    v = rand(n)
    duals = zeros(2, n)
    ndrange = (n, )
    AD._seed_kernel!(device)(duals, v, map; ndrange=ndrange)
    KA.synchronize(device)
    @test duals[2, map] == v

    # Partials extraction
    duals = rand(p+1, m)
    ## CSC
    ndrange = (m, ) # columns oriented
    AD.partials_kernel_csc!(device)(J.colptr, J.rowval, J.nzval, duals, colors; ndrange=ndrange)
    KA.synchronize(device)

    ## CSR
    Bp, Bj, Bx = ExaPF.convert2csr(J)
    ndrange = (n, ) # rows oriented
    AD.partials_kernel_csr!(device)(Bp, Bj, Bx, duals, colors; ndrange=ndrange)
    KA.synchronize(device)

    # Convert back to CSC and check results
    Ap = zeros(Int, m+1)
    Ai = zeros(Int, nnz(J))
    Ax = zeros(nnz(J))
    ExaPF.csr2csc(n, m, Bp, Bj, Bx, Ap, Ai, Ax)
    @test Ax == J.nzval
    return
end

function runtests(device, AT, SMT)
    for name_sym in names(@__MODULE__; all = true)
        name = string(name_sym)
        if !startswith(name, "test_")
            continue
        end
        test_func = getfield(@__MODULE__, name_sym)
        test_func(device, AT, SMT)
    end
end

end

