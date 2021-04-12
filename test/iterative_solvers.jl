using CUDA
using CUDA.CUSPARSE
using CUDAKernels
using ExaPF
using KernelAbstractions
using Krylov
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs

const LS = ExaPF.LinearSolvers

@testset "Iterative linear solvers with custom block Jacobi" begin
    # Square and preconditioned problems.
    function square_preconditioned(n :: Int=10)
        A   = ones(n, n) + (n-1) * Matrix(I, n, n)
        b   = 10.0 * [1:n;]
        M⁻¹ = 1/n * Matrix(I, n, n)
        return A, b, M⁻¹
    end
    A, b, N = square_preconditioned(100)
    x = similar(b); r = similar(b)
    spA = sparse(A)
    nblocks = 2
    precond = LS.BlockJacobiPreconditioner(spA, nblocks, CPU())
    LS.update(precond, spA, CPU())
    @testset "BICGSTAB" begin
        LS.ldiv!(LS.BICGSTAB(precond), x, spA, b)
        r = b - spA * x
        resid = norm(r) / norm(b)
        @test(resid ≤ 1e-6)
    end
    if has_cuda_gpu()
        cuspA = CuSparseMatrixCSR(spA)
        cux = CuVector(x)
        cub = CuVector(b)
        cuprecond = LS.BlockJacobiPreconditioner(cuspA, nblocks, CUDADevice())
        LS.update(cuprecond, cuspA, CUDADevice())
        @testset "BICGSTAB" begin
            LS.ldiv!(LS.BICGSTAB(precond), cux, cuspA, cub)
            cur = cub - cuspA * cux
            resid = norm(cur) / norm(cub)
            @test(resid ≤ 1e-6)
        end
    end
    # Embed preconditioner in linear solvers
    @testset "($LinSolver)" for LinSolver in ExaPF.list_solvers(CPU())
        algo = LinSolver(precond)
        LS.ldiv!(algo, x, spA, b)
        r = b - spA * x
        resid = norm(r) / norm(b)
        @test(resid ≤ 1e-6)
    end
end

@testset "Wrapping of iterative solvers" begin
    nblocks = 2
    n, m = 32, 32

    # Add a diagonal term for conditionning
    A = randn(n, m) + 15I
    x♯ = randn(m)
    b = A * x♯

    # Be careful: all algorithms work with sparse matrix
    A = sparse(A)
    # Init iterators for KernelAbstractions
    if has_cuda_gpu()
        iterators = zip([CPU(), CUDADevice()],
                        [Array{Float64}, CuVector{Float64}],
                        [SparseMatrixCSC, CuSparseMatrixCSR])
    else
        iterators = zip([CPU()], [Array{Float64}], [SparseMatrixCSC])
    end
    @testset "Run iterative solvers on device $(device)" for (device, V, SM) in iterators
        As = SM(A)
        bs = convert(V, b)
        x0 = convert(V, zeros(m))
        xs♯ = convert(V, x♯)

        # TODO: currently Preconditioner takes as input only sparse matrix
        # defined on the main memory.
        precond = LS.BlockJacobiPreconditioner(A, nblocks, device)

        # First test the custom implementation of BICGSTAB
        @testset "BICGSTAB" begin
            # Need to update preconditioner before resolution
            LS.update(precond, As, device)
            fill!(x0, 0.0)
            n_iters = LS.ldiv!(LS.BICGSTAB(precond), x0, As, bs)
            @test n_iters <= m
            @test x0 ≈ xs♯ atol=1e-6
        end
        @testset "Interface for iterative algorithm ($LinSolver)" for LinSolver in ExaPF.list_solvers(device)
            algo = LinSolver(precond)
            fill!(x0, 0.0)
            n_iters = LS.ldiv!(algo, x0, As, bs)
            @test n_iters <= m
            @test x0 ≈ xs♯ atol=1e-6
        end
    end
end
