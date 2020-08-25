using CUDA
using CUDA.CUSPARSE
using ExaPF
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs

@testset "Wrapping of iterative solvers" begin
    nblocks = 2
    n, m = 32, 32
    to = TimerOutputs.TimerOutput()

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
        precond = ExaPF.Precondition.Preconditioner(A, nblocks, device)

        # First test the custom implementation of BICGSTAB
        @testset "BICGSTAB" begin
            # Need to update preconditioner before resolution
            ExaPF.Precondition.update(As, precond, to)
            P = precond.P
            fill!(x0, 0.0)
            x_sol, n_iters = ExaPF.bicgstab(As, bs, P, x0, to)
            @test n_iters <= m
            @test x_sol ≈ xs♯ atol=1e-6
        end
        @testset "Interface for iterative algorithm ($algo)" for algo in ExaPF.list_solvers(device)
            fill!(x0, 0.0)
            n_iters = ExaPF.Iterative.ldiv!(x0, As, bs, algo, precond, to)
            @test n_iters <= m
            @test x0 ≈ xs♯ atol=1e-6
        end
    end
end
