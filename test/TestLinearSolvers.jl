module TestLinearSolvers

using Test

using LinearAlgebra
using Random
using SparseArrays

using KrylovPreconditioners
using ExaPF
const LS = ExaPF.LinearSolvers
const KP = KrylovPreconditioners

function generate_random_system(n::Int, m::Int)
    # Add a diagonal term for conditionning
    A = randn(n, m) + 15I
    x♯ = randn(m)
    b = A * x♯
    # Be careful: all algorithms work with sparse matrix
    spA = sparse(A)
    return spA, b, x♯
end

function test_preconditioners(device, AT, SMT)
    n, m = 100, 100
    A, b, x♯ = generate_random_system(n, m)
    # Transfer data to device
    A = A |> SMT
    b = b |> AT
    x♯ = x♯ |> AT
    x = similar(b); r = similar(b)
    nblocks = 2
    precond = BlockJacobiPreconditioner(A, nblocks, device)
    KP.update!(precond, A)
    Iₙ = Matrix{Float64}(I, n, n) |> AT
    Y = zeros(Float64, n, n) |> AT
    Y2 = zeros(Float64, n, n) |> AT
    mul!(Y, precond, Iₙ)
    for i=1:n
        eᵢ = Iₙ[:,i] |> AT
        mul!(view(Y2,:,i), precond, eᵢ)
    end
    @test Y ≈ Y2
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
    precond = BlockJacobiPreconditioner(A, nblocks, device)
    KP.update!(precond, A)
    linear_solver = LS.Bicgstab(A; P=precond)
    n_iters = LS.ldiv!(linear_solver, x, A, b)
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
    for olevel in [0, 1]
        precond = BlockJacobiPreconditioner(A; nblocks=nblocks, device=device, noverlaps=olevel)
        # Test printing
        println(devnull, precond)
        KP.update!(precond, A)
        @testset "Linear solver $LinearSolver" for LinearSolver in ExaPF.list_solvers(device)
            algo = LinearSolver(A; P=precond)
            fill!(x, 0.0)
            n_iters = LS.ldiv!(algo, x, A, b)
            @test n_iters <= m
            @test x ≈ x♯ atol=1e-6
        end
    end
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
