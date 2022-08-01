# Tests for custom kernels
module TestKernels

using LinearAlgebra
using Random
using SparseArrays
using Test

using ExaPF
using KernelAbstractions

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
    ev = ExaPF._transfer_fr_input!(device)(dest, src, mapping; ndrange=ndrange, dependencies=Event(device))
    wait(ev)
    @test dest == src[mapping]

    # TO
    src = rand(n)
    dest = zeros(m)
    ndrange = (n, )
    ev = ExaPF._transfer_to_input!(device)(dest, mapping, src; ndrange=ndrange, dependencies=Event(device))
    wait(ev)
    @test src == dest[mapping]
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
    ev = ExaPF.basis_kernel!(device)(output, vmag, vang, f, t, nl, nb; ndrange=ndrange, dependencies=Event(device))
    wait(ev)

    Δθ = vang[f] .- vang[t]
    vl2 = vmag[f] .* vmag[t]
    res = [vl2 .* cos.(Δθ);  vl2 .* sin.(Δθ); vmag .* vmag]
    @test res == output
end

function test_autodiff_kernel(device, AT, SMT)
    n, m = 10, 20
    J = sprandn(n, m, .2)
    colors = AD.SparseDiffTools.matrix_colors(J)
    p = length(unique(colors))

    # set_value!
    x = rand(n)
    duals = zeros(p+1, n)
    ndrange = (n, )
    ev = AD._set_value_kernel!(device)(duals, x; ndrange=ndrange, dependencies=Event(device))
    wait(ev)
    @test duals[1, :] == x

    # Seed with coloring for Jacobian/Hessian
    coloring = rand(1:p, n)
    duals = zeros(p+1, n)
    map = randperm(n)
    ndrange = (n, p)
    ev = AD._seed_coloring_kernel!(device)(duals, coloring, map; ndrange=ndrange, dependencies=Event(device))
    wait(ev)

    # Seed for Hessian-vector products
    v = rand(n)
    duals = zeros(2, n)
    ndrange = (n, )
    ev = AD._seed_kernel!(device)(duals, v, map; ndrange=ndrange, dependencies=Event(device))
    wait(ev)
    @test duals[2, map] == v

    # Partials extraction
    duals = rand(p+1, m)
    ## CSC
    ndrange = (m, ) # columns oriented
    ev = AD.partials_kernel_csc!(device)(J.colptr, J.rowval, J.nzval, duals, colors; ndrange=ndrange, dependencies=Event(device))
    wait(ev)

    ## CSR
    Bp, Bj, Bx = ExaPF.convert2csr(J)
    ndrange = (n, ) # rows oriented
    ev = AD.partials_kernel_csr!(device)(Bp, Bj, Bx, duals, colors; ndrange=ndrange, dependencies=Event(device))
    wait(ev)

    # Convert back to CSC and check results
    Ap = zeros(Int, m+1)
    Ai = zeros(Int, nnz(J))
    Ax = zeros(nnz(J))
    ExaPF.csr2csc(n, m, Bp, Bj, Bx, Ap, Ai, Ax)
    @test Ax == J.nzval
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

