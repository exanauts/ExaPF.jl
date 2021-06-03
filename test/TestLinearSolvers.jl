module TestLinearSolvers

using Test

using LinearAlgebra
using Random
using SparseArrays

using ExaPF
const LS = ExaPF.LinearSolvers

function generate_random_system(n::Int, m::Int)
    # Add a diagonal term for conditionning
    A = randn(n, m) + 15I
    x♯ = randn(m)
    b = A * x♯
    # Be careful: all algorithms work with sparse matrix
    spA = sparse(A)
    return spA, b, x♯
end

function test_custom_bicgstab(device, AT, SMT)
    n, m = 100, 100
    A, b, x♯  = generate_random_system(n, m)
    # Transfer data to device
    A = A |> SMT
    b = b |> AT
    x♯ = x♯ |> AT
    x = similar(b); r = similar(b)
    nblocks = 2
    precond = LS.BlockJacobiPreconditioner(A, nblocks, device)
    LS.update(precond, A, device)
    n_iters = LS.ldiv!(LS.BICGSTAB(A, precond), x, A, b)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ 1e-6)
    @test x ≈ x♯
    @test n_iters <= n
end

function test_all_linear_solvers(device, AT, SMT)
    n, m = 32, 32
    A, b, x♯  = generate_random_system(n, m)
    # Transfer data to device
    A = A |> SMT
    b = b |> AT
    x♯ = x♯ |> AT
    x = similar(b); r = similar(b)
    # Init preconditioner
    nblocks = 2
    precond = LS.BlockJacobiPreconditioner(A, nblocks, device)
    LS.update(precond, A, device)
    @testset "Linear solver $linear_solver" for linear_solver in ExaPF.list_solvers(device)
        algo = linear_solver(A, precond)
        fill!(x, 0.0)
        n_iters = LS.ldiv!(algo, x, A, b)
        @test n_iters <= m
        @test x ≈ x♯ atol=1e-6
    end
end

function runtests(device, AT, SMT)
    test_custom_bicgstab(device, AT, SMT)
    test_all_linear_solvers(device, AT, SMT)
end

end
